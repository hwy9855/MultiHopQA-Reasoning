import json
from utils import load_data, create_prompt, SYSTEM_PROMPTS
from tqdm import tqdm
import datasets

raw_data = load_data("data/musique_ans_v1.0_train.jsonl")

musique_train_qwen = []
for sample in tqdm(raw_data):
    prompt = create_prompt(sample)
    messages = [
    {"role": "system", "content": SYSTEM_PROMPTS["ft_qwen"]},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": sample["answer"]}
    ]
    musique_train_qwen.append({
        "id": sample["id"],
        "messages": messages
    })
    
with open("data/musique_train_qwen.jsonl", "w") as f:
    for sample in musique_train_qwen:
        f.write(json.dumps(sample))
        f.write("\n")

musique_train_llama = []
for sample in tqdm(raw_data):
    prompt = create_prompt(sample)
    messages = [
    {"role": "system", "content": SYSTEM_PROMPTS["ft_llama"]},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": sample["answer"]}
    ]
    musique_train_llama.append({
        "id": sample["id"],
        "messages": messages    
    })
    
with open("data/musique_train_llama.jsonl", "w") as f:
    for sample in musique_train_llama:
        f.write(json.dumps(sample))
        f.write("\n")

