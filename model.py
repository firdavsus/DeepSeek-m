import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

# ----- SwiGLU -----
class SwiGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

# ----- Rotary -----
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("n,d->nd", t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        cos = torch.cos(emb).unsqueeze(1).unsqueeze(0)  # [1, seq, 1, dim]
        sin = torch.sin(emb).unsqueeze(1).unsqueeze(0)
        return cos, sin

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    # x: [B, seq, heads, d_rope]
    return (x * cos) + (rotate_half(x) * sin)

# ----- Expert (fixed) -----
class Expert(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        # use 2*ffn_dim for SwiGLU: w1 -> 2*ffn_dim, then split inside SwiGLU -> ffn_dim
        self.w1 = nn.Linear(d_model, ffn_dim * 2)
        self.act = SwiGLU()
        self.w2 = nn.Linear(ffn_dim, d_model)
    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        return self.w2(x)

# ----- MLA (kept similar but consistent) -----
class MLA(nn.Module):
    def __init__(self, d_model, n_heads, d_rope, d_kv_comp):
        super().__init__()
        assert d_rope % 2 == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.split_dim = self.d_head - d_rope
        self.d_rope = d_rope

        self.W_dkv = nn.Linear(d_model, d_kv_comp)
        self.W_dq  = nn.Linear(d_model, d_kv_comp)

        self.W_uk = nn.Linear(d_kv_comp, n_heads * self.split_dim)
        self.W_uv = nn.Linear(d_kv_comp, n_heads * self.d_head)
        self.W_uq = nn.Linear(d_kv_comp, n_heads * self.split_dim)

        self.W_qr = nn.Linear(d_kv_comp, n_heads * d_rope)
        self.W_kr = nn.Linear(d_model, n_heads * d_rope)

        self.rotary = RotaryEmbedding(d_rope)
        self.output = nn.Linear(n_heads * self.d_head, d_model)

    def forward(self, h, past_kv=None):
        B, S, _ = h.shape
        # compress
        c_kv = self.W_dkv(h)
        k = self.W_uk(c_kv).view(B, S, self.n_heads, self.split_dim)
        v = self.W_uv(c_kv).view(B, S, self.n_heads, self.d_head)

        c_q = self.W_dq(h)
        q_base = self.W_uq(c_q).view(B, S, self.n_heads, self.split_dim)
        q_rot  = self.W_qr(c_q).view(B, S, self.n_heads, self.d_rope)

        cos, sin = self.rotary.get_cos_sin(S, device=h.device)
        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(self.W_kr(h).view(B, S, self.n_heads, self.d_rope), cos, sin)

        q = torch.cat([q_base, q_rot], dim=-1)  # [B,S,heads,d_head]
        k = torch.cat([k, k_rot], dim=-1)

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)  # [B,S,heads,d_head]
        out = out.contiguous().view(B, S, -1)
        return self.output(out), (c_kv, k_rot)

# ----- MoE (simple, correct) -----
class MoE(nn.Module):
    def __init__(self, d_model, n_experts, n_shared, top_k, ffn_dim):
        super().__init__()
        self.shared_experts = nn.ModuleList([Expert(d_model, ffn_dim) for _ in range(n_shared)])
        self.routed_experts = nn.ModuleList([Expert(d_model, ffn_dim) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)
        self.top_k = top_k
        self.n_experts = n_experts
        self.aux_loss = 0.0

    def forward(self, x):
        # shared part
        shared_out = sum(ex(x) for ex in self.shared_experts)

        # gating
        logits = self.gate(x)             # [B,S,n_experts]
        probs = F.softmax(logits, dim=-1) # [B,S,n_experts]
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # [B,S,top_k]

        routed_out = torch.zeros_like(x)
        # naive but correct dispatch
        B, S, _ = x.shape
        for k in range(self.top_k):
            idx_k = topk_idx[..., k]    # [B,S]
            prob_k = topk_probs[..., k] # [B,S]
            # for each expert, gather tokens
            for expert_i in range(self.n_experts):
                mask = (idx_k == expert_i)  # [B,S] bool
                if not mask.any():
                    continue
                xs = x[mask]                # [N, d_model]
                out = self.routed_experts[expert_i](xs)  # [N, d_model]
                routed_out[mask] += out * prob_k[mask].unsqueeze(-1)
        # simple expert balance aux (toy)
        counts = torch.zeros(self.n_experts, device=x.device)
        flat_idx = topk_idx.view(-1)
        ones = torch.ones_like(flat_idx, dtype=torch.float, device=x.device)
        counts.scatter_add_(0, flat_idx, ones)   # coarse
        self.aux_loss = counts.float().var()
        return shared_out + routed_out

# ----- Transformer block -----
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_experts, n_shared, top_k, dropout, n_heads, d_rope, d_kv_comp, ffn_dim):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = MLA(d_model, n_heads, d_rope, d_kv_comp)
        self.norm2 = nn.RMSNorm(d_model)
        self.moe = MoE(d_model, n_experts, n_shared, top_k, ffn_dim)
        self.dropout_mla = nn.Dropout(dropout)
        self.dropout_moe = nn.Dropout(dropout)

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv)
        x = x + self.dropout_mla(attn_out)
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout_moe(moe_out)
        return x, new_kv

# ----- AbsoluteTransformer (embed name 'embed' used consistently) -----
class AbsoluteTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_experts, config.n_shared, config.top_k,
                             config.dropout, config.n_heads, config.d_rope, config.d_kv_comp, config.ffn_dim)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        total_aux = 0.0
        for block in self.blocks:
            x, _ = block(x)
            total_aux = total_aux + block.moe.aux_loss
        return self.lm_head(self.norm(x)), total_aux
