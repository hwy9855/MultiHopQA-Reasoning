from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
import json
from tqdm import tqdm
import numpy as np
import torch
import os
from utils import create_random_prompt, load_data, SYSTEM_PROMPTS, compute_influence
import random
from utils import tokenize_for_eval, grouping_attention_score

def best_order(args):
    if args.exp_name:
        exp_name = f"{args.exp_name}-{args.K}"
    else:
        exp_name = f"{args.model_name.split('/')[-1]}-{args.K}"
    output_path = f"outputs/best_order-{exp_name}.json"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32, # Qwen2.5 models works unexpectedly with attention_implementation="eager" + torch.bfloat16
        device_map="auto",
        attn_implementation="eager", # For outputing attention weights
        token=args.token
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token)
    
    tokenizer_sup = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer_sup.add_tokens(["<instruct>", "<doc>", "<question>"])
    # Marking documents

    dataset = load_data(args.input_path)

    outputs = []
    if os.path.exists(output_path + ".tmp"):
        outputs = json.load(open(output_path + ".tmp"))

    dataset = dataset[len(outputs):]

    with torch.no_grad():
        for sample in tqdm(dataset):
            best_sample = {"max_ics_a": 0}
            all_samples = []
            n_docs = len(sample["paragraphs"])
            for _ in range(args.K):
                doc_order = list(range(n_docs))
                random.shuffle(doc_order)
                prompt = create_random_prompt(sample, doc_order=doc_order)
                prompt_sup = create_random_prompt(sample, doc_order=doc_order, with_special_token=True)
                messages = [
                {"role": "system", "content": SYSTEM_PROMPTS["answer_only_qwen"]},
                {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                messages_sup = [
                {"role": "system", "content": SYSTEM_PROMPTS["answer_only_qwen"]},
                {"role": "user", "content": prompt_sup}
                ]
                text_sup = tokenizer.apply_chat_template(
                    messages_sup,
                    tokenize=False,
                    add_generation_prompt=True
                )
                text += "\\boxed{"
                text_sup += "\\boxed{"

                special_ids, model_inputs = tokenize_for_eval(text, text_sup, tokenizer=tokenizer, tokenizer_sup=tokenizer_sup)
                if args.max_seq_len > 0:
                    if model_inputs.input_ids.shape[1] > args.max_seq_len:
                        continue
                output = model.generate(
                    **model_inputs.to(model.device),
                    max_new_tokens=512,
                    do_sample=False,
                    return_dict_in_generate=True, output_attentions=True
                )
                generated_ids = [
                output_ids[len(input_ids):].detach().cpu() for input_ids, output_ids in zip(model_inputs.input_ids, output.sequences)
                ]
                prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                grouped_scores, group_ids = grouping_attention_score(output.attentions, special_ids)
                if generated_ids[0][-2] == 92 and generated_ids[0].shape[0] > 2:
                    answer_tokens = [0, generated_ids[0].shape[0]-2]
                else:
                    answer_tokens = [0, generated_ids[0].shape[0]-1]
                _, ics_a = compute_influence(grouped_scores, n_docs, answer_tokens=answer_tokens, accumulate=True)
                max_ics_a = np.max(np.mean(ics_a[-1], axis=0))
                pred_sample = {
                        "id": sample["id"],
                        "max_ics_a": max_ics_a,
                        "doc_order": doc_order,
                        "prediction": prediction
                    }
                all_samples.append(pred_sample)
                if max_ics_a > best_sample["max_ics_a"]:
                    best_sample = pred_sample
                print(f"ref answer: {sample['answer']} \ndoc_order: {doc_order} \nics_a: {max_ics_a} \nprediction: {prediction}")
                del output
                torch.cuda.empty_cache()
            outputs.append({
                "best_sample": best_sample,
                "all_samples": all_samples
            })

            if len(outputs) % 10 == 0:
                json.dump(outputs, open(output_path + ".tmp", "w"))

    json.dump(outputs, open(output_path, "w"))
        
    
def main():
    parser = ArgumentParser(description="Example script using ArgumentParser")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='Name of the model to use')
    parser.add_argument('--exp_name', type=str, help='Name of the experiment')
    parser.add_argument('--input_path', type=str, default="data/musique_ans_v1.0_dev.jsonl", help='Path to the input text file')
    parser.add_argument('--K', type=int, default=20, help='Number of random doc orders to try')
    parser.add_argument('--max_seq_len', type=int, default=-1, help='Maximum sequence length, only set to positive value when met CUDA OOM. If set to positive, will skip all the samples that are longer than the max_seq_len after tokenization.')
    parser.add_argument('--token', type=str, default="HUGGINGFACE_TOKEN")

    args = parser.parse_args()
    best_order(args)

if __name__ == "__main__":
    main()
