from model import LLM, config
from cocnut import Coconut
import torch
import torch.nn.functional as F
from muon import Muon
from torch.cuda.amp import autocast, GradScaler
import csv
import random, math
from torch.nn.utils.rnn import pad_sequence
from torch import amp
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from time import time
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Data preparation
# ----------------------------
def evaluate(model, dataloader, pad_idx):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch_seq in dataloader:
            x = batch_seq[0][:, :-1].to(device)
            y = batch_seq[0][:, 1:].to(device)
            logits, aux_loss = model(x)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size),
                                   y.reshape(-1), ignore_index=pad_idx)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count

def get_unique_chars(path):
    unique_chars = set()

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for instr, inp, out, *_ in reader:
            # Combine all text fields
            text = instr + inp + out
            # Add each character to the set
            unique_chars.update(text)
    return unique_chars

def prepare_data_char(path, extra_tokens=None, val_ratio=0.01, stoi=None, itos=None):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Take first 25% for training (example)
    train_text = text
    val_len = int(len(train_text) * val_ratio)
    train_text, val_text = train_text[:-val_len], train_text[-val_len:]

    if stoi==None or itos==None:
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        print("new stoi and itos created!")

    if extra_tokens is not None:
        for tok in extra_tokens:
            if tok not in stoi:
                idx = len(stoi)
                stoi[tok] = idx
                itos[idx] = tok

        # Add unique chars from another dataset if needed
        unq = get_unique_chars("fine_tunin_data/alpaca.csv")
        if unq is not None:
            for tok in sorted(unq):
                if tok not in stoi:
                    idx = len(stoi)
                    stoi[tok] = idx
                    itos[idx] = tok

    config.vocab_size = len(stoi)
    print(f"Vocab size: {config.vocab_size}")

    data_train = torch.tensor([stoi.get(c, stoi.get('<unk>', 0)) for c in train_text], dtype=torch.long)
    data_val = torch.tensor([stoi.get(c, stoi.get('<unk>', 0)) for c in val_text], dtype=torch.long)

    return data_train, data_val, stoi, itos


def train(model, dataloader, dataloader_val, stoi, itos, epochs, print_each=500, coconut=False, accumulation_steps=8):
    if coconut:
        model.model.train().to(device)
    else:
        model.train().to(device)
    pad_idx = stoi['<pad>']

    hidden_weights = []
    hidden_gains_biases = []
    seen = set()
    if coconut:
        for block in model.model.blocks:
            for p in block.parameters():
                if not p.requires_grad: continue
                if id(p) in seen: continue
                seen.add(id(p))
                if p.ndim >= 2:
                    hidden_weights.append(p)
                else:
                    hidden_gains_biases.append(p)

        nonhidden_params = []
        for p in list(model.model.lm_head.parameters()) + list(model.model.token_emb.parameters()):
            if not p.requires_grad: continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            nonhidden_params.append(p)
    else:
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
            lr=0.01,
            momentum=0.6750,
            ns_steps=4,
            weight_decay=0.01
        ),
        dict(
            params=hidden_gains_biases + nonhidden_params,
            use_muon=False,
            lr=4e-5,
            betas=(0.9, 0.95),
            weight_decay=0.0
        ),
    ]
    scaler = amp.GradScaler() 
    optimizer_adam = Muon(param_groups)

    for epoch in range(epochs):
        current_next_checkpoint = 1
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        batch_start = time()
        for idx, batch_seq in enumerate(loop):
            batch_seq = batch_seq[0].to(device)  # [B, seq_len+1]
            x = batch_seq[:, :-1]
            y = batch_seq[:, 1:]

            with amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, aux_loss = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    y.reshape(-1),
                    ignore_index=pad_idx
                )
                aux_loss = torch.nan_to_num(aux_loss, nan=0.0, posinf=1e3, neginf=-1e3)
                aux_loss = aux_loss.clamp(-1e3, 1e3)
                loss += 0.0002 * aux_loss
                loss_for_backward = loss / accumulation_steps

            scaler.scale(loss_for_backward).backward()

            # Gradient accumulation
            do_step = ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(dataloader))
            if do_step:
                scaler.unscale_(optimizer_adam)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer_adam)
                scaler.update()
                optimizer_adam.zero_grad(set_to_none=True)

            # Print status
            if (idx + 1) % print_each == 0:
                print(f"cross -> Epoch {epoch} Loss: {loss.item():.4f} aux_loss: {aux_loss.item():.4f} --- {idx+1}/{len(dataloader)}")
                val_loss = evaluate(model, val_loader, pad_idx)
                print(f"[Val] Loss: {val_loss:.4f}")
                
                # Sample once per print checkpoint
                if coconut:
                    model.model.eval()
                else:
                    model.eval()
                with torch.no_grad():
                    prompt = "Good Morning Holms said "
                    context_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
                    output = model.sample(context_ids, max_new_tokens=200, temperature=1.0)[0]
                    generated_text = "".join([itos[i.item()] for i in output])
                    print(f"Sample: {generated_text}")
                if coconut:
                    model.model.train()
                else:
                    model.train()
                if coconut:
                    model.save_the_model(f"weights/model-coconut-{epoch}-{val_loss}.pt", stoi, itos)
                else:
                    torch.save(model.state_dict(), f"weights/model-simple-{epoch}-{val_loss}.pt")
                
                current_next_checkpoint += 1

            batch_time = time() - batch_start
            loop.set_postfix(loss=f"{loss.item():.4f}", aux=f"{aux_loss.item():.2f}", speed=f"{batch_time:.2f}s/batch")
            batch_start = time()
                
            if do_step:
                del logits, aux_loss, loss, loss_for_backward
                torch.cuda.empty_cache()
            
            if (idx + 1) % print_each == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"[GPU] Step {idx+1}: Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB")


        #example performance
        if coconut:
            model.model.eval()
        else:
            model.eval()
        with torch.no_grad():
            prompt = "Good Morning Holms said "
            context_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
            output = model.sample(context_ids, max_new_tokens=200, temperature=1.0)[0]
            generated_text = "".join([itos[i.item()] for i in output])
        if coconut:
            model.model.train()
        else:
            model.train()
        if coconut:
            model.save_the_model(f"weights/model-coconut-{epoch}.pt", stoi, itos)
        else:
            torch.save(model.state_dict(), f"weights/model-simple-{epoch}.pt")

def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Approximate size: {size_mb:.2f} MB")

def create_dataloader(data, seq_len, batch_size, pad_idx):
    # Pad at the beginning if data is too short
    if len(data) < seq_len + 1:
        pad_len = seq_len + 1 - len(data)
        pad_tensor = torch.full((pad_len,), pad_idx, dtype=torch.long)
        data = torch.cat([pad_tensor, data])  # pad at the beginning

    # Compute usable length (multiple of seq_len)
    num_sequences = (len(data) - 1) // seq_len
    usable_len = num_sequences * seq_len + 1
    data = data[-usable_len:]  # take the last usable_len tokens (rightmost)

    # Create sequences
    sequences = data.unfold(0, seq_len + 1, seq_len).contiguous()
    dataset = TensorDataset(sequences)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=6
    )
    return loader

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.reset_peak_memory_stats()

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    stoi = vocab['stoi']
    itos = vocab['itos']

    data, data_val, stoi, itos = prepare_data_char("data_text/combined_text.txt", extra_tokens=config.special_tokens, stoi=stoi, itos=itos)

    model_simple = LLM().to(device)
    model_simple.load_state_dict(torch.load('ready_model/model-simple-0-0.36046621447703875.pt'))
    print_model_info(model_simple)

    config.batch_size = 52
    pad_idx = stoi['<pad>']
    train_loader = create_dataloader(data, seq_len=config.max_len, batch_size=config.batch_size, pad_idx=stoi['<pad>'])
    val_loader   = create_dataloader(data_val, seq_len=config.max_len, batch_size=config.batch_size, pad_idx=stoi['<pad>'])

    train(model_simple, train_loader, val_loader, stoi, itos, epochs=2, accumulation_steps=4)
    torch.save(model_simple.state_dict(), "weights/model-simple-fin.pt")
    print("done, now with a coconut!")

    # # adding coconut
    # model = Coconut(
    #     model_simple, 
    #     num_latents=config.num_latents, 
    #     num_reasoning_steps=config.num_reasoning_steps, 
    #     top_k=config.top_k_samples, 
    #     inner_lr=5e-5
    # )
    # #model, stoi, itos = Coconut.load_the_model("weights/model-coconut-fin.pt", model)
    # model.stoi = stoi
    # config.batch_size = 16
    # dataloader = create_dataloader(data, seq_len=config.max_len, batch_size=config.batch_size, pad_idx=pad_idx)
    # train(model, train_loader, val_loader, stoi, itos, epochs=5, coconut=True, accumulation_steps=4)
    # model.save_the_model("weights/model-coconut-fin.pt", stoi, itos)

    # final model
    print("final model is ready!!!")