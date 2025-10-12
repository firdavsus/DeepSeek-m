from model import LLM, config
import torch
from cocnut import Coconut
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(input):
    with torch.no_grad():
        prompt = input
        context_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
        output = model.sample(context_ids, max_new_tokens=100, temperature=1.0, add_sos=False)[0] 
        generated_text = "".join([itos[i.item()] for i in output])
        print(f"generated: {generated_text}")

if __name__ == "__main__":
    # loading model
    model_simple = LLM().to(device)
    model = Coconut(
        model_simple, 
        num_latents=config.num_latents, 
        num_reasoning_steps=config.num_reasoning_steps, 
        top_k=config.top_k_samples, 
    )
    model, stoi, itos = Coconut.load_the_model("weights/model-coconut-4.pt", model)
    model.model.to(device)

    model.eval()
    while True:
        input_in = input("prompt: ")
        if input_in == "exit": break
        generate(input_in)