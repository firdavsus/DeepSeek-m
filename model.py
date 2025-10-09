import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class Config:
    def __init__(self):
        self.vocab_size=100
        self.embed_size = 64
        self.heads_size = 4
        self.num_layers = 2
        self.max_len = 100
        self.dropout = 0.1
        self.batch_size = 350
        self.d_rope = 8
        self.ff_hidden_mult=4
        self.ffn_dim = 48
        self.n_shared = 1
        self.n_experts = 2
        self.top_k_experts = 1
        self.d_kv_comp = 48
        self.num_latents=1
        self.num_reasoning_steps=1
        self.top_k_samples=40

config = Config()

# ---------- EXPERT ------------------
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(config.embed_size, config.ffn_dim*2)
        self.w2 = nn.Linear(config.ffn_dim, config.embed_size)
        self.act = SwiGLU()

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
    
class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_experts = nn.ModuleList([Expert() for _ in range(config.n_shared)])
        self.routed_experts = nn.ModuleList([Expert() for _ in range(config.n_experts)])
        self.gate = nn.Linear(config.embed_size, config.n_experts)

        nn.init.xavier_uniform_(self.gate.weight)
        self.aux_loss = 0.0

    def forward(self, x):
        # Shared experts process all tokens
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # Device-limited routing
        routed_logits = self.gate(x)
        probs = F.softmax(routed_logits, dim=-1)
        topk_probs, topk_indices = probs.topk(config.top_k_experts, dim=-1)

        # Expert balance loss
        expert_counts = torch.zeros(config.n_experts, device=x.device)
        expert_counts.scatter_add_(0, topk_indices.view(-1),
                                 torch.ones_like(topk_indices.view(-1), dtype=torch.float))
        #self.aux_loss += expert_counts.float().var() * 0.003  # α1 from paper

        expert_fraction = expert_counts / expert_counts.sum()
        balance_loss = expert_fraction.var() * 0.003

        mean_prob = probs.mean(dim=0)
        load_loss = (mean_prob * mean_prob.mean() / (mean_prob + 1e-9)).sum() * 1e-2

        self.aux_loss = balance_loss + load_loss

        # Sparse computation
        routed_out = torch.zeros_like(x)
        for k in range(config.top_k_experts):
            expert_mask = topk_indices[..., k]
            expert_contrib = torch.zeros_like(x)

            for expert_idx in range(config.n_experts):
                mask = (expert_mask == expert_idx)
                if mask.any():
                    expert_out = self.routed_experts[expert_idx](x[mask])
                    expert_contrib[mask] = expert_out * topk_probs[..., k][mask].unsqueeze(-1)

            routed_out += expert_contrib

        out = shared_out
        if config.top_k_experts != 0:
            out = shared_out + routed_out
        return out

# # ----- SwiGLU -----
class SwiGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale=40):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, 2).float() / (dim//2)))
        self.register_buffer("inv_freq", inv_freq)
        self.scale = 40

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    x_rot, x_base = x.split(cos.shape[-1], dim=-1)
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_rot, x_base], dim=-1)


class MLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_head = config.embed_size // config.heads_size
        self.split_dim = self.d_head - config.d_rope

        # Projections
        self.W_dkv = nn.Linear(config.embed_size, config.d_kv_comp)
        self.W_dq = nn.Linear(config.embed_size, config.d_kv_comp)

        # Changed value projection to use d_head instead of split_dim
        self.W_uk = nn.Linear(config.d_kv_comp, config.heads_size * self.split_dim)
        self.W_uv = nn.Linear(config.d_kv_comp, config.heads_size * self.d_head)
        self.W_uq = nn.Linear(config.d_kv_comp, config.heads_size * self.split_dim)

        self.W_qr = nn.Linear(config.d_kv_comp, config.heads_size * config.d_rope)
        self.W_kr = nn.Linear(config.embed_size, config.heads_size * config.d_rope)

        self.rotary = RotaryEmbedding(config.d_rope)
        self.output = nn.Linear(config.heads_size * self.d_head, config.embed_size)


        nn.init.xavier_uniform_(self.W_dkv.weight)
        nn.init.xavier_uniform_(self.W_dq.weight)
        nn.init.xavier_uniform_(self.W_uk.weight)
        nn.init.xavier_uniform_(self.W_uv.weight)
        nn.init.xavier_uniform_(self.W_uq.weight)
        nn.init.xavier_uniform_(self.W_qr.weight)
        nn.init.xavier_uniform_(self.W_kr.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, h, past_kv=None):
        batch_size, seq_len, _ = h.shape

        # KV Compression
        c_kv = self.W_dkv(h)
        k = self.W_uk(c_kv).view(batch_size, seq_len, config.heads_size, self.split_dim)
        v = self.W_uv(c_kv).view(batch_size, seq_len, config.heads_size, self.d_head)

        c_q = self.W_dq(h)
        q_base = self.W_uq(c_q).view(batch_size, seq_len, config.heads_size, self.split_dim)
        q_rot = self.W_qr(c_q).view(batch_size, seq_len, config.heads_size, config.d_rope)

        rotary_emb = self.rotary(seq_len)
        cos = torch.cos(rotary_emb).view(1, seq_len, 1, -1)  
        sin = torch.sin(rotary_emb).view(1, seq_len, 1, -1)

        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(
            self.W_kr(h).view(batch_size, seq_len, config.heads_size, config.d_rope),
            cos, sin
        )

        q = torch.cat([q_base, q_rot], dim=-1)
        k = torch.cat([k, k_rot], dim=-1)

        assert q.shape == k.shape, f"q/k shape mismatch: {q.shape} vs {k.shape}"
        assert q_base.shape[:-1] == q_rot.shape[:-1]
        assert q_base.size(-1) + q_rot.size(-1) == self.d_head

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=h.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)

        return self.output(out.contiguous().view(batch_size, seq_len, -1)), (c_kv, k_rot)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.embed_size)
        self.attn = MLA()
        self.norm2 = nn.RMSNorm(config.embed_size)
        self.moe = MoE()

    def forward(self, x, past_kv=None):
        # Attention with KV cache
        attn_out, new_kv = checkpoint(self.attn, self.norm1(x), past_kv, use_reentrant=False)
        x = x + attn_out

        # MoE with checkpointing
        moe_out = checkpoint(self.moe, self.norm2(x), use_reentrant=False)
        x = x + moe_out

        return x, new_kv
    
class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_size)
        self.blocks    = nn.ModuleList(
            [Block() for _ in range(config.num_layers)]
        )
        self.ln_f = nn.RMSNorm(config.embed_size)
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)
        nn.init.xavier_uniform(self.lm_head.weight)

    def forward(self, idx):
        x = self.token_emb(idx)
        total_aux_loss = 0.0

        for block in self.blocks:
            x, _ = block(x)
            total_aux_loss += block.moe.aux_loss
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss
    
    @torch.no_grad()
    def sample(self, context, max_new_tokens, temperature=1.0, top_k = None):
        for _ in range(max_new_tokens):
            # Crop context if longer than allowed
            idx_cond = context if context.size(1) <= self.token_emb.num_embeddings else context[:, -self.token_emb.num_embeddings:]
            logits, axu_loss = self(idx_cond)  # (1, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # Get last token's logits

            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float('-inf'))
                logits = mask.scatter(1, ix, v)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # Sample from probs
            context = torch.cat((context, next_token), dim=1)  # Append to sequence
        return context
    
