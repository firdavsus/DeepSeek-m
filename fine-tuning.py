import csv
import torch
from torch.nn.utils.rnn import pad_sequence
from muon import Muon
from torch.cuda.amp import autocast, GradScaler
from model import LLM, config
import torch.nn.functional as F
from cocnut import Coconut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(path, stoi):
    data = []
    masks = []

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for instr, inp, out, *_ in reader:
            seq = ['<sos>'] + list(instr) + list(inp) + ['<sos>'] + list(out) + ['<eos>']
            prefix_len = len(['<sos>'] + list(instr) + list(inp) + ['<sos>'])
            mask = [0 if i < prefix_len else 1 for i in range(len(seq))]

            if len(seq)>config.max_len:
                continue
            seq = [stoi.get(ch, stoi['<unk>']) for ch in seq]

            data.append(torch.tensor(seq, dtype=torch.long))
            masks.append(torch.tensor(mask, dtype=torch.bool))
    print(len(data))
    return data, masks


def get_sequential_batches_random(data, masks, batch_size, device, pad_idx):
    indices = torch.randperm(len(data))
    data = [data[i] for i in indices]
    masks = [masks[i] for i in indices]

    for i in range(0, len(data), batch_size):
        batch_seq = data[i:i + batch_size]
        batch_mask = masks[i:i + batch_size]

        batch_seq = pad_sequence(batch_seq, batch_first=True, padding_value=pad_idx)
        batch_mask = pad_sequence(batch_mask, batch_first=True, padding_value=0)

        yield batch_seq.to(device), batch_mask.to(device)


## train loop----------------------------------------------
def train(model, data, masks, stoi, itos, epochs, print_each=1000, accumulation_steps=8):
    model.train().to(device)

    # build param groups (your code)
    hidden_weights = []
    hidden_gains_biases = []
    seen = set()
    for block in model.model.blocks:
        for p in block.parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.ndim >= 2:
                hidden_weights.append(p)
            else:
                hidden_gains_biases.append(p)

    nonhidden_params = []
    for p in list(model.model.lm_head.parameters()) + list(model.model.token_emb.parameters()):
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        nonhidden_params.append(p)

    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=0.0001, weight_decay=0.01),
        dict(params=hidden_gains_biases + nonhidden_params, use_muon=False, lr=2e-5, betas=(0.9, 0.95), weight_decay=0.01),
    ]

    scaler = GradScaler()
    optimizer_adam = Muon(param_groups)

    pad_idx = stoi['<pad>']
    total_batches = (len(data) + config.batch_size - 1) // config.batch_size

    for epoch in range(epochs):
        batches = get_sequential_batches_random(data, masks, config.batch_size, device, pad_idx=pad_idx)

        # zero once to start accumulation
        optimizer_adam.zero_grad(set_to_none=True)

        for idx, (single, mask) in enumerate(batches):
            single = single.to(device)
            mask = mask.to(device)

            with autocast():
                logits, aux_loss = model(single[:, :-1])        # [B, T-1, V]
                targets = single[:, 1:]                         # [B, T-1]
                mask_tgt = mask[:, 1:]                          # align with targets

                loss_per_token = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                    ignore_index=pad_idx
                )
                mask_float = mask_tgt.reshape(-1).float()
                loss = (loss_per_token * mask_float).sum() / mask_float.sum().clamp(min=1)
                loss = loss + 0.0005 * aux_loss

                # normalize by accumulation steps
                loss_for_backward = loss / accumulation_steps

            # scale & backward
            scaler.scale(loss_for_backward).backward()

            # If it's time to step (or final leftover), unscale, clip, step, update, zero grads
            do_step = ((idx + 1) % accumulation_steps == 0)
            # check if final batch and gradients leftover
            is_last = (idx + 1) == total_batches
            if do_step or is_last:
                # unscale before clipping
                scaler.unscale_(optimizer_adam)   # unscale grads for this optimizer
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # step + update via scaler (works even if optimizer is Muon)
                scaler.step(optimizer_adam)
                scaler.update()

                # zero for next accumulation cycle
                optimizer_adam.zero_grad(set_to_none=True)

            # Logging: show the un-normalized loss for readability
            if idx % print_each == 0:
                # loss_for_backward is scaled-down; show original loss
                print_loss = float(loss_for_backward.item() * accumulation_steps)
                print(f"Epoch {epoch} Step {idx}/{total_batches} Loss: {print_loss:.4f} aux_loss: {aux_loss.item():.6f}")

                model.eval()
                with torch.no_grad():
                    prompt = "What is the Capital of France?" + '<sos>'
                    context_ids = torch.tensor([stoi.get(c, stoi['<unk>']) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
                    output = model.sample(context_ids, max_new_tokens=50, temperature=1.0)[0]
                    generated_text = "".join([itos[i.item()] for i in output])
                    print("Sample:", generated_text)
                    prompt = "What is 2+2=?" + '<sos>'
                    context_ids = torch.tensor([stoi.get(c, stoi['<unk>']) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
                    output = model.sample(context_ids, max_new_tokens=50, temperature=1.0)[0]
                    generated_text = "".join([itos[i.item()] for i in output])
                    print("Sample:", generated_text)
                model.train()

        # end epoch
        print(f"Epoch {epoch} finished.")
        model.save_the_model(f"weights/model-tuned-{epoch}.pt", stoi, itos)



if __name__ == "__main__":
    model_simple = LLM().to(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = Coconut(
        model_simple, 
        num_latents=config.num_latents, 
        num_reasoning_steps=config.num_reasoning_steps, 
        top_k=config.top_k_samples, 
        inner_lr=6e-6
    )
    config.batch_size = 64
    model, stoi, itos = Coconut.load_the_model("weights/model-coconout-fin.pt", model)
    data, masks = get_data("fine_tunin_data/alpaca.csv", stoi)
    train(model, data, masks, stoi, itos, epochs=20)

    # final model
    print("final fine-tuned model is ready!!!")