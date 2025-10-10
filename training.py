from model import LLM, config
from cocnut import Coconut
import torch
import torch.nn.functional as F
from muon import Muon
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Data preparation
# ----------------------------
def prepare_data_char(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    text = text[:len(text)//2]

    config.vocab_size = len(stoi)
    print(f"Vocab size: {config.vocab_size}")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos

def get_sequential_batches_random(data, batch_size, seq_len):
    num_sequences = (data.size(0) - 1) // seq_len
    usable_len = num_sequences * seq_len + 1
    data = data[:usable_len]
    sequences = data.unfold(0, seq_len + 1, seq_len)  
    
    indices = torch.randperm(sequences.size(0))
    sequences = sequences[indices]

    for i in range(0, sequences.size(0), batch_size):
        yield sequences[i:i + batch_size].clone()

def train(model, data, stoi, itos, epochs, print_each=1000, coconut=False):
    model.train().to(device)

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
            if id(p) in seen:
                continue
            seen.add(id(p))
            nonhidden_params.append(p)

    param_groups = [
        dict(params=hidden_weights, use_muon=True,
            lr=0.002, weight_decay=0.01),
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
            lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
    ]
    scaler = GradScaler()
    optimizer_adam = Muon(param_groups)

    for epoch in range(epochs):
        batch = get_sequential_batches_random(data, config.batch_size, config.max_len)

        for idx, single in enumerate(batch):
            single=single.to(device)
            optimizer_adam.zero_grad()
            #optimizer_muon.zero_grad()
            with torch.cuda.amp.autocast():
                logits, aux_loss = model(single[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    single[:, 1:].reshape(-1)
                )
                loss += 0.0005 * aux_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_adam.step()

            if idx%print_each==0:
                print(f"Epoch {epoch} Loss: {loss.item():.4f} aux_loss: {aux_loss.item():.4f} --- {idx}/{len(data)//config.batch_size}")
                model.eval()
                with torch.no_grad():
                    prompt = "Good Morning Holms said "
                    context_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
                    output = model.sample(context_ids, max_new_tokens=50, temperature=1.0)[0] 
                    generated_text = "".join([itos[i.item()] for i in output])
                    print(f"loss: {loss.item():.4f} aux_loss: {aux_loss.item():.4f} Sample: {generated_text}")
                model.train()

        print(f"Epoch {epoch} Loss: {loss.item():.4f} aux_loss: {aux_loss.item():.4f}")

        #example performance
        model.eval()
        with torch.no_grad():
            prompt = "Good Morning Holms said "
            context_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
            output = model.sample(context_ids, max_new_tokens=50, temperature=1.0)[0] 
            generated_text = "".join([itos[i.item()] for i in output])
            print(f"loss: {loss.item():.4f} aux_loss: {aux_loss.item():.4f} Sample: {generated_text}")
        model.train()
        if coconut:
            model.save_the_model(f"weights/model-coconut-{epoch}.pt", stoi, itos)
        else:
            torch.save(model.state_dict(), f"weights/model-simple-{epoch}.pt")

if __name__ == "__main__":
    data, stoi, itos = prepare_data_char("summary.txt")

    model_simple = LLM().to(device)

    config.batch_size = 400
    train(model_simple, data, stoi, itos, epochs=7)
    torch.save(model_simple.state_dict(), "weights/model-simple-fin.pt")
    print("done, now with a coconut!")

    # adding coconut
    model = Coconut(
        model_simple, 
        num_latents=config.num_latents, 
        num_reasoning_steps=config.num_reasoning_steps, 
        top_k=config.top_k_samples, 
        inner_lr=5e-5
    )
    config.batch_size = 256
    train(model, data, stoi, itos, epochs=13, coconut=True)
    model.save_the_model("weights/model-cocnout-fin.pt", stoi, itos)

    # final model
    print("final model is ready!!!")