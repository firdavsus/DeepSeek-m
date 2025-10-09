from model import LLM, config
import torch
from cocnut import Coconut
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(input):
    with torch.no_grad():
        prompt = input
        context_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
        output = model.sample(context_ids, max_new_tokens=50, temperature=1.0)[0] 
        generated_text = "".join([itos[i.item()] for i in output])
        print(f"generated: {generated_text}")

if __name__ == "__main__":
    # loading model
    model_simple = LLM().load_state_dict(torch.load("model-coconut.pt")).to(device)
    model = Coconut(
        model_simple, 
        num_latents=config.num_latents, 
        num_reasoning_steps=config.num_reasoning_steps, 
        top_k=config.top_k_samples, 
    )

    #loading config
    with open("stoi.pkl", "rb") as f:
        stoi = pickle.load(f)

    with open("itos.pkl", "rb") as f:
        itos = pickle.load(f)

    model.eval()
    while True:
        input = input("prompt: ")
        if input == "exit": break
        generate(input)