from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import os


def train_bpe_tokenizer(corpus_path, save_dir="tokenizer", vocab_size=3000, min_frequency=2):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # Basic normalizer + byte-level pre-tokenizer (robust for arbitrary text)
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size,
                         min_frequency=min_frequency,
                         special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.train([corpus_path], trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(os.path.join(save_dir, "bpe-tokenizer.json"))
    return tokenizer

from model import LLM, config
from cocnut import Coconut
import torch
import torch.nn.functional as F
import math
from tokenizers import Tokenizer
import os

def load_or_train_tokenizer(corpus_path="summary.txt", save_dir="tokenizer", vocab_size=16000):
    tok_path = os.path.join(save_dir, "bpe-tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        tokenizer = train_bpe_tokenizer(corpus_path, save_dir=save_dir, vocab_size=vocab_size)
    return tokenizer

def prepare_data_bpe(path, tokenizer, device=torch.device("cpu")):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    enc = tokenizer.encode(text)
    ids = enc.ids 

    data = torch.tensor(ids, dtype=torch.long).to(device)
    vocab_size = tokenizer.get_vocab_size()
    return data, tokenizer, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch._dynamo.config.suppress_errors = True

def train(epochs, model, data, tokenizer, device):
    model = model.to(device)
    model.train()

    # steps / schedulers
    num_sequences = (len(data) - 1) // config.max_len
    batches_per_epoch = math.ceil(num_sequences / config.batch_size)
    total_steps = epochs * batches_per_epoch

    base_muon_lr = 0.002
    base_adam_lr = 3e-4   # <-- correct LR for AdamW

    # For now we use a single AdamW for simplicity; if you enable Dion, group params properly
    adam_opt = torch.optim.AdamW(model.parameters(), lr=base_adam_lr, weight_decay=0.01)

    scheduler_adam = torch.optim.lr_scheduler.OneCycleLR(
        adam_opt, max_lr=base_adam_lr, total_steps=total_steps, pct_start=0.1
    )

    use_cuda = torch.cuda.is_available()
    autocast_device = "cuda" if use_cuda else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    global_step = 0

    for epoch in range(epochs):
        train_on_data = get_sequential_batches_from_token_ids(data, config.batch_size, config.max_len)
        total_batches = (len(data) + config.batch_size * config.max_len - 1) // (config.batch_size * config.max_len)
        print_each = max(1, math.ceil(total_batches * 0.25))

        for idx_curr, inputs in enumerate(train_on_data):
            # inputs is (batch_size, seq_len+1) on CPU — move to device
            inputs = inputs.to(device)

            # --- clear grads ---
            adam_opt.zero_grad(set_to_none=True)

            # --- forward ---
            with torch.amp.autocast(device_type=autocast_device, enabled=use_cuda):
                logits, aux_loss = model(inputs[:, :-1])
                logits_flat = logits.reshape(-1, config.vocab_size)
                targets_flat = inputs[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean', label_smoothing=0.1)
                # if model doesn't produce aux_loss set aux_loss=0.0
                loss = loss + 0.0001 * (aux_loss if aux_loss is not None else 0.0)

            # --- backward ---
            scaler.scale(loss).backward()

            scaler.unscale_(adam_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(adam_opt)
            scaler.update()

            scheduler_adam.step()

            global_step += 1

            # optional periodic sampling (every print_each)
            if idx_curr % print_each == 0:
                model.eval()
                with torch.no_grad():
                    prompt = "My name is "
                    enc = tokenizer.encode(prompt)
                    context_ids = enc.ids
                    context = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(device)
                    output = model.sample(context, max_new_tokens=50)  # shape (1, L)
                    out_ids = output[0].tolist()
                    out_ids_list = out_ids[0].tolist() if isinstance(out_ids, torch.Tensor) else out_ids

                    text_full = tokenizer.decode(out_ids_list, skip_special_tokens=True)
                    text_full=text_full.replace("Ġ", " ")

                    print(f"loss: {loss.item():.4f} aux_loss: {aux_loss.item()} Sample: {text_full}")
                model.train()

        # end of epoch logging
        print(f"Epoch {epoch} Loss: {loss.item():.4f} aux_loss: {(aux_loss.item() if hasattr(aux_loss, 'item') else 0.0):.4f}")

        # checkpointing
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"weights/model-{loss.item():.3f}.pth")

def prepare_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chars = sorted(list(set(text)))

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    config.vocab_size = len(stoi)
    print(len(stoi))

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    return data, stoi, itos

def get_sequential_batches_from_token_ids(data, batch_size, seq_len):
    num_sequences = (data.size(0) - 1) // seq_len
    usable_len = num_sequences * seq_len + 1
    data = data[:usable_len]
    sequences = data.unfold(0, seq_len + 1, seq_len)  
    for i in range(0, sequences.size(0), batch_size):
        batch = sequences[i:i + batch_size].clone()
        yield batch 

def go():
    tokenizer = train_bpe_tokenizer("summary.txt", save_dir="tokenizer", vocab_size=16000)

    data, tokenizer, vocab_size = prepare_data_bpe("summary.txt", tokenizer, device=device)
    config.vocab_size = vocab_size

    model = LLM()
    # model.load_state_dict(torch.load("weights/model-end.pth"))

    train(20, model, data, tokenizer, device=device)

    torch.save(model.state_dict(), "weights/model-end.pth")

if __name__ == "__main__":
    go()
