from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from bi_models import Qwen2ForCausalLMInf, LlamaForCausalLMInf
from argparse import ArgumentParser
import json
from tqdm import tqdm
import numpy as np
import torch
import pickle as pk
import os
from utils import create_prompt, load_data, SYSTEM_PROMPTS

def inference(args):
    if args.exp_name:
        exp_name = f"{args.exp_name}-{args.mode}" + (f"-{args.doc_order}" if args.doc_order else "-ori")
    else:
        exp_name = f"{args.model_name.split('/')[-1]}-{args.mode}" + (f"-{args.doc_order}" if args.doc_order else "-ori")
    output_path = f"outputs/{exp_name}"
    output_path += ".json"

    if args.bi_attn and args.model_type == "qwen":
        model = Qwen2ForCausalLMInf.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
            token=args.token
        ).eval()
    elif args.bi_attn and args.model_type == "llama":
        model = LlamaForCausalLMInf.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
            token=args.token
        ).eval()
    elif args.model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            device_map="auto",
            token=args.token
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token)
    if args.model_type == "t5":
        tokenizer.model_max_length = 8192

    dataset = load_data(args.data_path)

    outputs = []
    if os.path.exists(output_path + ".tmp"):
        outputs = json.load(open(output_path + ".tmp", "r"))


    dataset = dataset[len(outputs):]

    with torch.no_grad():
        for sample in tqdm(dataset):
            if args.model_type == "t5":
                prompt = create_prompt(sample, order=args.doc_order)
                text = prompt
            else:
                prompt = create_prompt(sample, order=args.doc_order)
                messages = [
                {"role": "system", "content": SYSTEM_PROMPTS[args.mode]},
                {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                if "answer_only" in args.mode:
                    text += "\\boxed{"

            model_inputs = tokenizer(text, return_tensors="pt")

            output = model.generate(
                **model_inputs.to(model.device),
                max_new_tokens=768,
                do_sample=False,
                return_dict_in_generate=True
            )
            if args.model_type == "t5":
                generated_ids = output.sequences
            else:
                generated_ids = [
                output_ids[len(input_ids):].detach().cpu() for input_ids, output_ids in zip(model_inputs.input_ids, output.sequences)
                ]
            prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            outputs.append({
                "id": sample["id"],
                "prediction": prediction
            })

            if len(outputs) % 10 == 0:
                json.dump(outputs, open(output_path + ".tmp", "w"))


    res = {
        "predictions": outputs
    }
    
    json.dump(res, open(output_path, "w"))
        
    
def main():
    parser = ArgumentParser(description="Example script using ArgumentParser")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='Name of the model to use')
    parser.add_argument('--model_type', choices=["qwen", "llama", "t5"], required=True)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--data_path', type=str, default="data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument('--mode', choices=["answer_only_qwen", "answer_only_llama", "cot", "ft_qwen", "ft_llama", "t5"], required=True)
    parser.add_argument('--bi_attn', action="store_true", help="Use bi-directional attention")
    parser.add_argument('--doc_order', type=str, default=None, help="doc order to use, e.g. backward, forward, forward_0, ..., forward_5. Leave blank for original order.")
    parser.add_argument('--token', type=str, default="HUGGINGFACE_TOKEN")
    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    main()
