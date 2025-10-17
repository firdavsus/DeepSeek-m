from datasets import load_dataset
import pandas as pd

from huggingface_hub import login

login()
# Load the dataset
dataset = load_dataset("BAAI/Infinity-Instruct", "7M_core", trust_remote_code=True)

train_data = dataset['train']

# Prepare lists to store data
questions = []
answers = []
limit = 100000

for idx, item in enumerate(train_data):
    q = item['conversations']
    # Combine best_answer + all possible answers
    # combined_answer = item['best_answer'] + " | " + " | ".join(item['possible_answers'])
    if item["langdetect"]!="en":
        continue
    
    questions.append(q[0]['value'])
    answers.append(q[1]['value'])

df = pd.DataFrame({
    'question': questions,
    'answers': answers
})

# Save to CSV
df.to_csv("truthfulqa_combined_answers.csv", index=False)
print("CSV saved!")

# import pandas as pd

# # === CONFIG ===
# path = "fine_tunin_data/alpaca.csv"  # <-- your CSV path
# output_path = "cleaned.csv"  # where cleaned CSV will be saved
# col1, col2, col3 = 0, 1, 2  # indexes of the 3 columns (0-based)

# # === LOAD CSV ===
# df = pd.read_csv(path)

# # === IDENTIFY VALID ROWS ===
# def is_valid(row):
#     c1 = str(row.iloc[col1]).strip()
#     c2 = str(row.iloc[col2]).strip()
#     c3 = str(row.iloc[col3]).strip()

#     both_empty = (c1 == "" or c1.lower() == "nan") and (c2 == "" or c2.lower() == "nan")
#     third_empty = (c3 == "" or c3.lower() == "nan")
#     both_empty = (c1=="#NAME?" or c2=="#NAME?" or c3=="#NAME?")

#     return not (both_empty or third_empty)

# valid_mask = df.apply(is_valid, axis=1)

# # === FILTER DATAFRAME ===
# df_cleaned = df[valid_mask].reset_index(drop=True)

# # === REPORT ===
# removed_count = len(df) - len(df_cleaned)
# print(f"Total rows: {len(df)}")
# print(f"Removed invalid rows: {removed_count}")
# print(f"Cleaned rows remaining: {len(df_cleaned)}")

# # === SAVE CLEANED CSV ===
# df_cleaned.to_csv(output_path, index=False)
# print(f"Cleaned CSV saved as {output_path}")
