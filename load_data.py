from datasets import load_dataset
import os

os.makedirs("data_text", exist_ok=True)

bookcorpus = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

# book_part = bookcorpus.shuffle(seed=42).select(range(int(0.25 * len(bookcorpus))))

book_text = " ".join(bookcorpus["text"])

combined_text = book_text

# --- Save ---
output_path = "data_text/combined_text.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(combined_text)

print("Books saved! 25%")
print(f"🎉 Combined text saved at {output_path}")
