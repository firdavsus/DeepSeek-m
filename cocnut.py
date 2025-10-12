import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from model import config, LLM

class Coconut(nn.Module):
    def __init__(self, transformer, num_latents=3, num_reasoning_steps=3, inner_lr=0.001, top_k=50):
        super().__init__()
        self.top_k=top_k
        self.model = transformer
        self.num_latents = num_latents
        self.num_reasoning_steps = num_reasoning_steps

        d = transformer.token_emb.embedding_dim

        self.latents = nn.Parameter(torch.randn(num_latents, d))
        nn.init.xavier_uniform_(self.latents)
        self.sot = nn.Parameter(torch.zeros(1, d))
        self.eot = nn.Parameter(torch.zeros(1, d))
        self.stoi = None
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
        aux_losses = []
    
        for block in self.model.blocks:
            x, _, aux_loss = block(x)  # block.forward now returns x, new_kv, aux_loss
            aux_losses.append(aux_loss)
    
        x = self.model.ln_f(x)
        logits = self.model.lm_head(x)
        total_aux_loss = sum(aux_losses) / len(aux_losses)
        return logits, total_aux_loss

    def forward(self, idx):
        INNER_GRAD_MAX_NORM = 0.5  
        LATENT_CLAMP = 3.0    
        INNER_LR_MAX = 1e-3      
        LATENT_L2_REG = 1e-5  

        device = idx.device
        B, T = idx.shape

        prompt_embed = self.model.token_emb(idx)   # [B, T, D]

        latents_base = repeat(self.latents, "n d -> b n d", b=B).to(device)  # [B, n, D]
        sot = repeat(self.sot, "1 d -> b 1 d", b=B).to(device)              # [B, 1, D]
        eot = repeat(self.eot, "1 d -> b 1 d", b=B).to(device)              # [B, 1, D]

        latents = latents_base.clone().detach().requires_grad_(True)  # leaf tensor with grads

        for step in range(self.num_reasoning_steps): 
            # ... inside your forward, per step:
            logits_all, _ = self._run_blocks_on_embeddings(sot, latents, prompt_embed, eot=eot)
            logits_prompt = logits_all[:, 1 + self.num_latents : 1 + self.num_latents + T, :]  # [B, T, V]

            if T <= 1:
                inner_loss = torch.tensor(0.0, device=device, dtype=logits_prompt.dtype)
            else:
                inner_logits = logits_prompt[:, :-1, :].reshape(-1, logits_prompt.size(-1))  # [B*(T-1), V]
                inner_targets = idx[:, 1:].reshape(-1)  # [B*(T-1)]
                inner_loss = F.cross_entropy(inner_logits, inner_targets, reduction='mean')

            inner_loss = inner_loss + LATENT_L2_REG * latents.pow(2).mean()

            inner_loss = torch.nan_to_num(inner_loss, nan=0.0, posinf=1e3, neginf=-1e3)

            grads = torch.autograd.grad(inner_loss, latents, create_graph=True, allow_unused=True)[0]
            if grads is None:
                grads = torch.zeros_like(latents)

            grads = torch.nan_to_num(grads, nan=0.0, posinf=1e3, neginf=-1e3)

            B = grads.shape[0]
            grads_view = grads.view(B, -1)
            grad_norm = grads_view.norm(dim=1).clamp_min(1e-6)  # [B]
            scale = (INNER_GRAD_MAX_NORM / grad_norm).clamp(max=1.0)  # [B]
            scale = scale.view(B, *([1] * (grads.dim()-1)))  # broadcast shape back to grads
            grads = grads * scale

            raw_lr = self.inner_lrs[step] if self.inner_lrs.numel() > 1 else self.inner_lrs[0]
            lr = F.softplus(raw_lr)             # ensures >0
            lr = lr.clamp(max=INNER_LR_MAX)     # limit magnitude

            latents = latents - lr * grads

            latents = torch.clamp(latents, -LATENT_CLAMP, LATENT_CLAMP)

        logits_all_final, aux_loss = self._run_blocks_on_embeddings(sot, latents, prompt_embed, eot=eot)
        logits_prompt_final = logits_all_final[:, 1 + self.num_latents : 1 + self.num_latents + T, :]  # [B, T, V]
        aux_loss = torch.nan_to_num(aux_loss, nan=0.0, posinf=1e3, neginf=-1e3)
        logits_prompt_final = torch.nan_to_num(logits_prompt_final , nan=0.0, posinf=1e3, neginf=-1e3)
        return logits_prompt_final, aux_loss
    
    def save_the_model(self, path, stoi, itos):
    # coconut_state should contain only coconut-specific params (no 'model.' keys)
        full_state = self.state_dict()
        coconut_state = {k: v for k, v in full_state.items() if not k.startswith("model.")}
        checkpoint = {
            "transformer_state": self.model.state_dict(),
            "coconut_state": coconut_state,
            "stoi": stoi,
            "itos": itos
        }
        torch.save(checkpoint, path)
    
    @staticmethod
    def load_the_model(path, coconut):
        checkpoint = torch.load(path, map_location="cpu")
        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]

        if coconut.model.token_emb.num_embeddings != len(stoi):
            config.vocab_size = len(stoi)
            coconut.model = LLM()

        coconut.model.load_state_dict(checkpoint["transformer_state"])
        # load only coconut keys back
        coconut_state = checkpoint["coconut_state"]
        coconut.load_state_dict({k: v for k, v in coconut_state.items()}, strict=False)
        coconut.stoi = stoi
        return coconut, stoi, itos

    def sample(self, context, max_new_tokens=20, temperature=1.0, run_inner_at_inference=True, add_sos=False):
        # NOTE: do NOT use @torch.no_grad() decorator on this method
        self.model.eval()
        device = context.device
        B, T = context.shape

        latents = repeat(self.latents, "n d -> b n d", b=B).to(device)
        sot = repeat(self.sot, "1 d -> b 1 d", b=B).to(device)
        eot = repeat(self.eot, "1 d -> b 1 d", b=B).to(device)
        prompt_embed = self.model.token_emb(context)

        if run_inner_at_inference and self.num_reasoning_steps > 0:
            # Enable gradient computation for inner optimization (but only here)
            with torch.enable_grad():
                # start with latents that require grad
                latents = latents.clone().detach().requires_grad_(True)

                for step in range(self.num_reasoning_steps):
                    logits_all, _ = self._run_blocks_on_embeddings(sot, latents, prompt_embed, eot=eot)
                    logits_prompt = logits_all[:, 1 + self.num_latents : 1 + self.num_latents + T, :]
                    if T <= 1:
                        break

                    inner_logits = logits_prompt[:, :-1, :].reshape(-1, logits_prompt.size(-1))
                    inner_targets = context[:, 1:].reshape(-1)
                    inner_loss = F.cross_entropy(inner_logits, inner_targets)

                    # compute grads wrt latents (no create_graph -> no higher-order graph)
                    grads = torch.autograd.grad(inner_loss, latents, create_graph=False, allow_unused=True)[0]
                    if grads is None:
                        grads = torch.zeros_like(latents)

                    # get lr (tensor), make scalar
                    raw_lr = self.inner_lrs[step] if self.inner_lrs.numel() > 1 else self.inner_lrs[0]
                    lr = F.softplus(raw_lr).clamp(max=1e-3)  # or your config

                    # update latents WITHOUT tracking this update in the graph
                    with torch.no_grad():
                        latents.sub_(lr * grads)           # in-place update (efficient)
                        # optional clamp to keep values stable
                        latents.clamp_(-3.0, 3.0)

                    # re-enable grad for the next inner iteration
                    latents.requires_grad_(True)

        # Now do generation under no_grad (efficient)
        out_tokens = [context[:, i:i+1] for i in range(T)]
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = torch.cat(out_tokens, dim=1)
                if add_sos:
                    sos_id = self.stoi['<sos>']
                    idx_cond = torch.tensor([[sos_id]], dtype=torch.long, device=device)
                logits, _ = self.model(idx_cond)
                last_logits = logits[:, -1, :] / max(temperature, 1e-8)
                if self.top_k is not None:
                    v, ix = torch.topk(last_logits, self.top_k)
                    mask = torch.full_like(last_logits, float('-inf'))
                    last_logits = mask.scatter(1, ix, v)
                probs = F.softmax(last_logits, dim=-1)
                # protect against NaNs / zero-sum
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                sums = probs.sum(dim=-1, keepdim=True)
                zero_rows = (sums == 0).squeeze(-1)
                if zero_rows.any():
                    probs[zero_rows] = 1.0 / float(last_logits.size(-1))
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token == self.stoi["<eos>"]:
                    break
                out_tokens.append(next_token)

        self.model.train()
        return torch.cat(out_tokens, dim=1)
