import csv
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from model import LLM, config
import torch.nn.functional as F
import random
import pickle
from tqdm.auto import tqdm  # <- added tqdm
from torch import amp
import matplotlib.pyplot as plt
from muon import Muon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv.field_size_limit(10_000_000)

@torch.no_grad()
def evaluate(model, data, masks, stoi):
    model.eval()
    pad_idx = stoi['<pad>']
    total_loss = 0.0
    total_tokens = 0

    dataloader = get_dataloader(data, masks, config.batch_size, pad_idx=pad_idx, shuffle=False)
    total_val_batches = (len(data) + config.batch_size - 1) // config.batch_size

    for single, mask in tqdm(dataloader, total=total_val_batches, desc="Evaluate", leave=False):
        single = single.to(device)
        mask = mask.to(device)

        logits, aux_loss = model(single[:, :-1])
        targets = single[:, 1:]
        mask_tgt = mask[:, 1:]

        loss_per_token = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            targets.reshape(-1),
            reduction="none",
            ignore_index=pad_idx
        )

        mask_float = mask_tgt.reshape(-1).float()
        loss = (loss_per_token * mask_float).sum()
        total_loss += loss.item()
        total_tokens += mask_float.sum().item()

    model.train()
    return total_loss / max(total_tokens, 1)
    
def _tokenize(text, stoi):
    tokens = []
    i = 0
    L = len(text)

    # Sort special tokens by length (so longer ones like <instruction_start> are matched first)
    special_tokens = sorted(config.special_tokens, key=len, reverse=True)

    while i < L:
        matched = False
        for tok in special_tokens:
            if text.startswith(tok, i):
                tokens.append(stoi.get(tok, stoi.get("<unk>", 0)))
                i += len(tok)
                matched = True
                break

        if not matched:
            tokens.append(stoi.get(text[i], stoi.get("<unk>", 0)))
            i += 1

    return tokens


def get_data(path, stoi):
    data = []
    masks = []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)

        for inp, out in reader:
            # build full sequence string with special tokens (single token each)
            seq_str = "<sos>## User: " + inp + "\n## Agent: " + out + "<eos>"
            seq_str = seq_str.replace("�", "<unk>")

            # tokenize whole sequence (special tokens handled by _tokenize)
            seq_tokens = _tokenize(seq_str, stoi)

            # compute prefix tokens length (everything up to the start of the answer)
            prefix_str = "<sos>## User: " + inp + "\n## Agent: "
            prefix_tokens = _tokenize(prefix_str, stoi)
            prefix_len = len(prefix_tokens)

            # mask: 0 for prefix (do not predict), 1 for answer tokens (predict)
            mask = [0 if i < prefix_len else 1 for i in range(len(seq_tokens))]

            if len(seq_tokens) > config.max_len:
                continue

            data.append(torch.tensor(seq_tokens, dtype=torch.long))
            masks.append(torch.tensor(mask, dtype=torch.bool))
    print(len(data))
    return data, masks

# --- simple Dataset + dataloader collate_fn ---
class SeqDataset(Dataset):
    def __init__(self, sequences, masks):
        self.sequences = sequences
        self.masks = masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.masks[idx]

def _collate_fn_factory(pad_idx):
    def collate_fn(batch):
        seqs, masks = zip(*batch)
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
        return seqs_padded, masks_padded
    return collate_fn

def get_dataloader(data, masks, batch_size, pad_idx, shuffle=True, num_workers=0):
    ds = SeqDataset(data, masks)
    collate_fn = _collate_fn_factory(pad_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

## train loop----------------------------------------------
def train(model, data, val_data, masks, val_masks, stoi, itos, epochs, print_each=200, accumulation_steps=2):
    model.train().to(device)

    train_losses = []
    val_losses = []

    hidden_weights = []
    hidden_gains_biases = []
    seen = set()

    scaler = GradScaler()
    for block in model.blocks:
        for p in block.parameters():
            if not p.requires_grad: continue
            if id(p) in seen: continue
            seen.add(id(p))
            if p.ndim >= 2:
                hidden_weights.append(p)
            else:
                hidden_gains_biases.append(p)

    nonhidden_params = []
    for p in list(model.lm_head.parameters()) + list(model.token_emb.parameters()):
        if not p.requires_grad: continue
        if id(p) in seen: continue
        seen.add(id(p))
        nonhidden_params.append(p)

    param_groups = [
        dict(
            params=hidden_weights,
            use_muon=True,
            lr=0.005,
            momentum=0.5750,
            ns_steps=4,
            weight_decay=0.01
        ),
        dict(
            params=hidden_gains_biases + nonhidden_params,
            use_muon=False,
            lr=3e-5,
            betas=(0.9, 0.95),
            weight_decay=0.0
        ),
    ]

    pad_idx = stoi['<pad>']
    total_batches = (len(data) + config.batch_size - 1) // config.batch_size

    optimizer_adam = Muon(param_groups)

    for epoch in range(epochs):
        dataloader = get_dataloader(data, masks, config.batch_size, pad_idx=pad_idx, shuffle=True)

        # zero once to start accumulation
        optimizer_adam.zero_grad(set_to_none=True)

        # tqdm progress bar for this epoch
        pbar = tqdm(dataloader, total=total_batches, desc=f"Epoch {epoch}", leave=False)
        for idx, (single, mask) in enumerate(pbar):
            single = single.to(device)
            mask = mask.to(device)

            with amp.autocast(device_type=device.type, dtype=torch.bfloat16):
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
                loss = loss + 0.0001 * aux_loss

                # normalize by accumulation steps
                loss_for_backward = loss / accumulation_steps

            # scale & backward
            scaler.scale(loss_for_backward).backward()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "aux": f"{aux_loss.item():.4f}"
            })

            # If it's time to step (or final leftover), unscale, clip, step, update, zero grads
            do_step = ((idx + 1) % accumulation_steps == 0)
            # check if final batch and gradients leftover
            is_last = (idx + 1) == total_batches
            if do_step or is_last and idx!=0:
                scaler.unscale_(optimizer_adam)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer_adam)
                scaler.update()
                optimizer_adam.zero_grad(set_to_none=True)

                # zero for next accumulation cycle
                optimizer_adam.zero_grad(set_to_none=True)

            # Logging: show the un-normalized loss for readability
            if idx % print_each==0 and idx!=0:
                # loss_for_backward is scaled-down; show original loss
                print_loss = float(loss_for_backward.item() * accumulation_steps)
                print(f"Epoch {epoch} Step {idx}/{total_batches} Loss: {print_loss:.4f} aux_loss: {aux_loss.item():.6f}")

                model.eval()
                with torch.no_grad():
                    # first sample
                    prompt_text = "<sos>## User: What is the Capital of France?\n## Agent:"
                    context_ids = torch.tensor([_tokenize(prompt_text, stoi)], dtype=torch.long).to(device)
                    output = model.sample(context_ids, max_new_tokens=50, temperature=1.0)[0]
                    generated_text = "".join([itos[i.item()] for i in output])
                    print("Sample:", generated_text)

                    # second sample
                    prompt_text = "<sos>## User: 2+2=?\n## Agent:"
                    context_ids = torch.tensor([_tokenize(prompt_text, stoi)], dtype=torch.long).to(device)
                    output = model.sample(context_ids, max_new_tokens=50, temperature=1.0)[0]
                    generated_text = "".join([itos[i.item()] for i in output])
                    print("Sample:", generated_text)
                model.train()
                val_loss = evaluate(model, val_data, val_masks, stoi)
                print(f"Epoch {epoch} validation loss: {val_loss:.4f}")
                torch.save(model.state_dict(), f"weights/model-tuned-{epoch}-{val_loss:.4f}.pt")
                train_losses.append(print_loss)
                val_losses.append(val_loss)
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label='Train Loss', color='blue')
                plt.plot(val_losses, label='Validation Loss', color='orange')
                plt.title('Training and Validation Loss')
                plt.xlabel('Print Step')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('loss_curve.png', dpi=300)
                plt.close()
                print("📉 Saved loss graph to loss_curve.png")

            if do_step:
                del logits, aux_loss, loss, loss_for_backward
                torch.cuda.empty_cache()

        pbar.close()

        # end epoch
        print(f"Epoch {epoch} finished.")
        torch.save(model.state_dict(), f"weights/model-tuned-{epoch}-end-{val_loss:.4f}.pt")
        val_loss = evaluate(model, val_data, val_masks, stoi)
        print(f"Epoch {epoch} validation loss: {val_loss:.4f}")

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

stoi = vocab['stoi']
itos = vocab['itos']

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    model_simple = LLM().to(device)
    model_simple.load_state_dict(torch.load('model-tuned-1-0.5481.pt'))
    # model = Coconut(
    #     model_simple,
    #     num_latents=config.num_latents,
    #     num_reasoning_steps=config.num_reasoning_steps,
    #     top_k=config.top_k_samples,
    #     inner_lr=6e-6
    # )
    data, masks = get_data("truthfulqa_combined_answers.csv", stoi)
    config.batch_size = 2
    combined = list(zip(data, masks))
    random.shuffle(combined)

    data, masks = zip(*combined)

    split_idx = int(len(data) * 0.998)
    train_data, val_data = data[:split_idx], data[split_idx:]
    train_masks, val_masks = masks[:split_idx], masks[split_idx:]
    # model, stoi, itos = Coconut.load_the_model("weights/model-coconout-fin.pt", model)
    train(model_simple, train_data, val_data, train_masks, val_masks, stoi, itos, epochs=8, accumulation_steps=32)

    # final model
    print("final fine-tuned model is ready!!!")


