import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class Coconut(nn.Module):
    def __init__(self, transformer, num_latents=3, num_reasoning_steps=3, inner_lr=0.001, top_k=50):
        super().__init__()
        self.top_k=top_k
        self.model = transformer
        self.num_latents = num_latents
        self.num_reasoning_steps = num_reasoning_steps

        d = transformer.token_emb.embedding_dim

        self.latents = nn.Parameter(torch.randn(num_latents, d))
        self.sot = nn.Parameter(torch.zeros(1, d))
        self.eot = nn.Parameter(torch.zeros(1, d))
        nn.init.normal_(self.sot, std=0.02)
        nn.init.normal_(self.eot, std=0.02)

        if isinstance(inner_lr, float) or isinstance(inner_lr, int):
            self.inner_lrs = nn.Parameter(torch.full((num_reasoning_steps,), float(inner_lr)))
        else:
            v = torch.tensor(inner_lr, dtype=torch.float)
            assert v.numel() == num_reasoning_steps
            self.inner_lrs = nn.Parameter(v)

    def _run_blocks_on_embeddings(self, sot, latents, prompt_embed, eot=None):
        x = torch.cat([sot, latents, prompt_embed] + ([] if eot is None else [eot]), dim=1)  # [B, L, D]
        for block in self.model.blocks:
            # block(x) returns (x, new_kv)
            x, _ = block(x)
        x = self.model.ln_f(x)
        logits_all = self.model.lm_head(x)  # [B, L, V]

        # collect aux_loss (if blocks accumulate them). Convert python floats -> tensors
        aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.model.blocks:
            if hasattr(block, "moe") and hasattr(block.moe, "aux_loss"):
                aux_loss = aux_loss + torch.tensor(getattr(block.moe, "aux_loss", 0.0), device=x.device)
        return logits_all, aux_loss

    def forward(self, idx):
        device = idx.device
        B, T = idx.shape

        prompt_embed = self.model.token_emb(idx)   # [B, T, D]

        latents_base = repeat(self.latents, "n d -> b n d", b=B).to(device)  # [B, n, D]
        sot = repeat(self.sot, "1 d -> b 1 d", b=B).to(device)              # [B, 1, D]
        eot = repeat(self.eot, "1 d -> b 1 d", b=B).to(device)              # [B, 1, D]

        latents = latents_base.clone().detach().requires_grad_(True)  # leaf tensor with grads

        for step in range(self.num_reasoning_steps):
            logits_all, _ = self._run_blocks_on_embeddings(sot, latents, prompt_embed, eot=None)
            # logits_all shape: [B, 1 + n + T, V]; prompt logits occupy positions [1+n : 1+n+T)
            logits_prompt = logits_all[:, 1 + self.num_latents : 1 + self.num_latents + T, :]  # [B, T, V]

            if T <= 1:
                inner_loss = torch.tensor(0.0, device=device, dtype=logits_prompt.dtype)
            else:
                inner_logits = logits_prompt[:, :-1, :].reshape(-1, logits_prompt.size(-1))  # [B*(T-1), V]
                inner_targets = idx[:, 1:].reshape(-1)  # [B*(T-1)]
                inner_loss = F.cross_entropy(inner_logits, inner_targets)

            # compute gradients w.r.t. latents (only). create_graph=True so we can backprop through inner updates
            grads = torch.autograd.grad(inner_loss, latents, create_graph=True, retain_graph=True)[0]

            # step size for this inner iteration (learnable)
            lr = self.inner_lrs[step] if self.inner_lrs.numel() > 1 else self.inner_lrs[0]

            # gradient descent update on latents (we keep latents as a differentiable tensor)
            latents = latents - lr * grads
            latents = torch.clamp(latents, -10, 10)  # <- stability clamp
            latents = latents.detach().requires_grad_(True)  # see below
            # ensure next latents is a differentiable node (it already is), keep requires_grad True

        logits_all_final, aux_loss = self._run_blocks_on_embeddings(sot, latents, prompt_embed, eot=eot)
        logits_prompt_final = logits_all_final[:, 1 + self.num_latents : 1 + self.num_latents + T, :]  # [B, T, V]

        return logits_prompt_final, aux_loss

    @torch.no_grad()
    def sample(self, context, max_new_tokens=20, temperature=1.0, run_inner_at_inference=False):
        device = context.device
        B, T = context.shape

        latents = repeat(self.latents, "n d -> b n d", b=B).to(device)
        sot = repeat(self.sot, "1 d -> b 1 d", b=B).to(device)
        prompt_embed = self.model.token_emb(context)

        if run_inner_at_inference:
            # do inner steps in eval mode but without gradient accumulation
            latents = latents.clone().detach().requires_grad_(True)
            for step in range(self.num_reasoning_steps):
                logits_all, _ = self._run_blocks_on_embeddings(sot, latents, prompt_embed, eot=None)
                logits_prompt = logits_all[:, 1 + self.num_latents : 1 + self.num_latents + T, :]
                if T <= 1:
                    break
                inner_logits = logits_prompt[:, :-1, :].reshape(-1, logits_prompt.size(-1))
                inner_targets = context[:, 1:].reshape(-1)
                inner_loss = F.cross_entropy(inner_logits, inner_targets)
                grads = torch.autograd.grad(inner_loss, latents, create_graph=False)[0]
                lr = self.inner_lrs[step] if self.inner_lrs.numel() > 1 else self.inner_lrs[0]
                latents = (latents - lr * grads).detach()  # detach to avoid storing graphs in sampling

        out_tokens = [context[:, i:i+1] for i in range(T)]
        for _ in range(max_new_tokens):
            idx_cond = torch.cat(out_tokens, dim=1)  # [B, cur_len]
            maybe = self.model(idx_cond)
            if isinstance(maybe, tuple):
                logits, _ = maybe
            else:
                logits = maybe
            last_logits = logits[:, -1, :] / temperature
            if self.top_k is not None:
                v, ix = torch.topk(last_logits, self.top_k)
                mask = torch.full_like(last_logits, float('-inf'))
                last_logits = mask.scatter(1, ix, v)
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out_tokens.append(next_token)
        return torch.cat(out_tokens, dim=1)
