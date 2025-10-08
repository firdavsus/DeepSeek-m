from model import AbsoluteTransformer
from cocnut import Coconut
import torch
import torch.nn.functional as F
import math
from muon import Muon

class Config:
    def __init__(self):
        self.vocab_size = 100
        self.d_model = 256
        self.n_layers = 6
        self.n_heads = 4
        self.d_kv_comp = 64
        self.d_rope = 16
        self.n_experts = 4
        self.n_shared = 2
        self.top_k = 2
        self.max_len = 100
        self.ffn_dim = 64
        self.device_groups = 0
        self.num_latents = 2
        self.num_reasoning_steps = 2
        self.dropout=0.1
        self.batch_size = 64

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, model, data, stoi, itos, device):
    model = model.to(device)
    model.train()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # normal rule for Muon
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name and "expert" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    num_sequences = (len(data) - 1) // config.max_len
    batches_per_epoch = math.ceil(num_sequences / config.batch_size)
    total_steps = epochs * batches_per_epoch

    muon_opt = Muon(muon_params, lr=0.01, rank=0, world_size=1)
    adam_opt  = torch.optim.AdamW(adam_params, lr=3e-4, weight_decay=0.1)

    scheduler_muon = torch.optim.lr_scheduler.OneCycleLR(
        muon_opt, max_lr=0.01, total_steps=total_steps, pct_start=0.1
    )
    scheduler_adam = torch.optim.lr_scheduler.OneCycleLR(
        adam_opt,  max_lr=3e-4, total_steps=total_steps, pct_start=0.1
    )

    use_cuda = torch.cuda.is_available()
    autocast_device = "cuda" if use_cuda else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    global_step = 0

    for epoch in range(epochs):
        train_on_data = get_sequential_batches(data, config.batch_size, config.max_len)
        total_batches = (len(data) + config.batch_size * config.max_len - 1) // (config.batch_size * config.max_len)
        print_each = math.ceil(total_batches*0.1)
        for idx_curr, inputs in enumerate(train_on_data):
            inputs = inputs.to(device)

            # --- clear grads BEFORE forward/backward ---
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_cuda):
                logits, aux_loss = model(inputs[:, :-1])
                logits_flat = logits.reshape(-1, config.vocab_size)
                targets_flat = inputs[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
                loss += 0.0001 * aux_loss

            scaler.scale(loss).backward()

            has_muon_grads = any((p.grad is not None) for p in muon_params)
            has_adam_grads  = any((p.grad is not None) for p in adam_params)

            if has_muon_grads:
                scaler.unscale_(muon_opt)
            if has_adam_grads:
                scaler.unscale_(adam_opt)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            if has_muon_grads:
                scaler.step(muon_opt)
            if has_adam_grads:
                scaler.step(adam_opt)

            scaler.update()

            if has_muon_grads:
                scheduler_muon.step()
            if has_adam_grads:
                scheduler_adam.step()

            global_step += 1

            if idx_curr % print_each == 0:
                print(f"Current Loss: {loss.item():.4f}")
                model.eval()
                with torch.no_grad():
                    prompt = "Good Morning Holms said "
                    context = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
                    output = model.sample(context, max_new_tokens=50)
                    generated_text = "".join([itos[i.item()] for i in output[0]])
                    print(f"Sample:\n{generated_text}\n")
                model.train()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            prompt = "Good Morning Holms said "
            context = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
            output = model.sample(context, max_new_tokens=50)
            generated_text = "".join([itos[i.item()] for i in output[0]])
            print(f"Sample after epoch {epoch}:\n{generated_text}\n")
        model.train()

def prepare_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chars = sorted(list(set(text)))

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    config.vocab_size = len(stoi)

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    return data, stoi, itos

def get_sequential_batches(data, batch_size, seq_len):
    num_sequences = (len(data) - 1) // seq_len
    data = data[:num_sequences * seq_len + 1]
    sequences = data.unfold(0, seq_len + 1, seq_len)
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        yield batch 

def go():
    data, stoi, itos = prepare_data("summary.txt")
    transformer = AbsoluteTransformer(config)

    model = Coconut(transformer, num_latents=config.num_latents, num_reasoning_steps=config.num_reasoning_steps)

    train(20, model, data, stoi, itos, device=device)

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    go()

''' main optimizations:

    -hard:
    +Multi-Head Latent Attention (MLA)
    +MoE

    -average diff:
    +RoPE pos enc 
    +Latent thinking (coconut paper)

    -ez ones:
    +Muno optimizer
    +swiGLU
    +RMSNorm
    +initialization
'''