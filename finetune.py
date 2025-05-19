from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM

from peft import LoraConfig
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM
)

import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                      help="Model ID to use for training")
    parser.add_argument("--lora_r", type=int, default=8,
                      help="Rank of LoRA approximation")
    parser.add_argument("--lora_alpha", type=int, default=16,
                      help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                      help="LoRA dropout probability")
    parser.add_argument("--output_dir", type=str, default="finetuned_models/qwen_7b",
                      help="Directory to save model checkpoints")
    parser.add_argument("--data_path", type=str, default="data/musique_train_qwen.jsonl",
                      help="Path to training data file")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                      help="Batch size per GPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                      help="Batch size per GPU for evaluation")
    parser.add_argument("--token", type=str, default="HUGGINGFACE_TOKEN",
                      help="Token for the model")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                      help="Number of training epochs")
    return parser.parse_args()

def finetune(args):
    model_name = args.model_name
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    output_dir = args.output_dir
    data_path = args.data_path
    token = args.token
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            ],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    response_template = "<|im_start|>assistant"
    if model_name.startswith("meta-llama"):
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        response_template = "<|start_header_id|>assistant<|end_header_id|>"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype="auto",
        token=token
    )

    dataset_train = load_dataset("json", data_files=data_path, split="train")

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, response_template=response_template, mlm=False
    )


    training_args = SFTConfig(output_dir=output_dir, 
                            bf16=True,
                            per_device_train_batch_size=1,
                            per_device_eval_batch_size=1,
                            max_seq_length=4096,
                            save_total_limit=3,
                            num_train_epochs=args.num_train_epochs,
                            seed=42
                            )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset_train,
        args=training_args,
        peft_config=lora_config,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)

if __name__ == "__main__":
    args = parse_args()
    finetune(args)
