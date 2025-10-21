from model import LLM, config
import torch
import pickle
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

stoi = vocab['stoi']
itos = vocab['itos']

def generate(input_prompt):
    with torch.no_grad():
        context_ids = sample_with_prompt(model, stoi, itos, prompt=input_prompt, max_len=1024, temperature=1.0, top_k=150)
        generated_text = "".join([itos[i.item()] for i in context_ids])
        print(f"A: {generated_text}")

def sample_with_prompt(model, stoi, itos, prompt="", temperature=1.0, top_k=None, max_len=512):
    prompt = "## User: " + prompt + "\n## Agent: "
    device = next(model.parameters()).device
    sos_id = stoi["<sos>"]
    eos_id = stoi["<eos>"]

    prompt_ids = [stoi.get(ch, stoi["<unk>"]) for ch in prompt]
    context = torch.tensor([[sos_id] + prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_len - len(context[0])):
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float('-inf'))
            logits = mask.scatter(1, ix, v)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        context = torch.cat((context, next_token), dim=1)

        if next_token.item() == eos_id:
            break

    return context.squeeze(0)

if __name__ == "__main__":
    # loading model
    model = LLM().to(device)
    model.load_state_dict(torch.load('weights/model-tuned-1-0.5481.pt'))

    model.to(device)

    model.eval()
    while True:
        input_in = input("Q: ")
        if input_in == "exit": break
        generate(input_in)