from model import LLM, config
import torch
from cocnut import Coconut
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

stoi = vocab['stoi']
itos = vocab['itos']

def generate(input):
    with torch.no_grad():
        prompt = input
        context_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
        output = model.sample(context_ids, max_new_tokens=150, temperature=1.0, top_k=40)[0] 
        generated_text = "".join([itos[i.item()] for i in output])
        print(f"generated: {generated_text}")

def sample_with_prompt(model, stoi, itos, prompt="", temperature=1.0, top_k=None, max_len=512):
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
    model.load_state_dict(torch.load('weights/model-simple-0-0.43909742001331215.pt'))
    # model = Coconut(
    #     model_simple, 
    #     num_latents=config.num_latents, 
    #     num_reasoning_steps=config.num_reasoning_steps, 
    #     top_k=config.top_k_samples, 
    # )
    # model, stoi, itos = Coconut.load_the_model("weights/model-coconout-fin.pt", model)
    model.to(device)

    model.eval()
    while True:
        input_in = input("prompt: ")
        if input_in == "exit": break
        generate(input_in)