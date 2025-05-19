# MultiHopQA-Reasoning

This is the official repository for the accepted ACL 2025 main paper "Masking in Multi-hop QA: An Analysis of How Language Models Perform with Context Permutation". This repository contains code for evaluating and analyzing language models' performance on multi-hop question answering tasks under different context ordering settings.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hwy9855/MultiHopQA-Reasoning.git
cd MultiHopQA-Reasoning
```

2. Create a conda environment and install the required dependencies:
```bash
conda create -n multihopqa python=3.8
conda activate multihopqa
pip install -r requirements.txt
```

3. Download the data:
Download the MusiQue following the instructions from the [official repository](https://github.com/StonyBrookNLP/musique). Put the `musique_ans_v1.0_dev.jsonl` and `musique_ans_v1.0_train.jsonl` files in the `data/` directory. You can also use other datasets following the same format.

## Experiments

### Inference with context permutation

To run inference with different context ordering strategies:

```bash
python inference.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --model_type "qwen" \ # qwen, llama, t5
    --exp_name "qwen_7b_musique" \
    --data_path "data/musique_ans_v1.0_dev.jsonl" \
    --mode "answer_only_qwen" \ # answer_only_qwen, answer_only_llama, cot, ft_qwen, ft_llama, t5
    --doc_order "forward" \ # forward, backward, forward_0, ..., forward_5
    --token "hf_..." 
```

For reproducing the results in the paper, you can use the following command:

#### Flan-T5-xl
```bash
bash scripts/eval_t5_xl.sh
```

#### Qwen 2.5-7B
```bash
bash scripts/eval_qwen_7b.sh # answer only and CoT
```

#### Llama 3.1-8B
```bash
bash scripts/eval_llama_8b.sh # answer only and CoT
```

Remember to replace the `hf_...` with your own Hugging Face token.

### Evaluation

You can use the following command to compute the metrics for the predictions:

```bash
python src/scripts/evaluate.py \
    --pred_path "predictions.json" \
    --ref_path "data/musique_ans_v1.0_dev.jsonl"
```

For evaluating CoT predictions from Qwen 2.5 models, you should specify `--cot` to capture the final answer from the CoT. Noted that since small Qwen2.5 models (<3B) and Llama models do not following the CoT format well, we treat the last line of the generation as the final answer.

### Fine-tuning

To replicate our experiments on finetuned models, you can first run the following command to finetune a Qwen 2.5 model or Llama 3.x model with causal mask:

```bash
python finetune.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --model_type "qwen" \ # qwen, llama
    --exp_name "qwen_7b_musique" \
    --data_path "data/musique_ans_v1.0_dev.jsonl" \
    --mode "answer_only_qwen" \ # answer_only_qwen, answer_only_llama, cot, ft_qwen, ft_llama, t5
    --doc_order "forward" \ # forward, backward, forward_0, ..., forward_5
    --token "hf_..." 
```

For finetuning Qwen 2.5 model or Llama 3.x model, use the `finetune_bi.py` script instead:

```bash
python finetune_bi.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --model_type "qwen" \ # qwen, llama
    --exp_name "qwen_7b_musique" \
    --data_path "data/musique_ans_v1.0_dev.jsonl" \
    --mode "answer_only_qwen" \ # answer_only_qwen, answer_only_llama, cot, ft_qwen, ft_llama, t5
```

We also provide the finetune scripts for replicate the experiment in the paper.

```bash
bash scripts/finetune_qwen_7b.sh # finetune with causal mask
bash scripts/finetune_qwen_7b_bi.sh # finetune with bi-directional attention
bash scripts/finetune_llama_8b.sh # finetune with causal mask
bash scripts/finetune_llama_8b_bi.sh # finetune with bi-directional attention
```

You can then use the finetuned models to run inference with the same command as above, or use the provided scripts to run inference with the finetuned models:

```bash
bash scripts/eval_qwen_7b_finetuned.sh # inference with finetuned Qwen 2.5 7B model
bash scripts/eval_qwen_7b_finetuned_bi.sh # inference with finetuned Qwen 2.5 7B model with bi-directional attention
bash scripts/eval_llama_8b_finetuned.sh # inference with finetuned Llama 3.1 8B model
bash scripts/eval_llama_8b_finetuned_bi.sh # inference with finetuned Llama 3.1 8B model with bi-directional attention
```

### Best Order Evaluation

In this study, we found that LLMs are able to determine the best context ordering for multi-hop QA tasks with the peak Information Contribution (IC) score. You can replicate the experiments for this part by running the following command:

```bash
python best_order.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --input_path "data/musique_ans_v1.0_dev.jsonl" \
    --K 20
```

Noted that outputing attention weights requires setting `attn_implementation` to `eager` in the model configuration, which takes much more memory. If you met CUDA OOM, you can set the `--max_seq_len` to a positive value to skip the samples that are longer than the max_seq_len after tokenization. It is NOT recommended to use `fp16` and `bf16` combined with `attn_implementation="eager"`, which will lead to notable performance decrease.

After generating the predictions, you can use the following command to evaluate the best order:

```bash
python best_order_eval.py \
    --pred_path "predictions.json" \
    --ref_path "data/musique_ans_v1.0_dev.jsonl"
```
## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{huang-etal-2025-masking,
  title={Masking in Multi-hop QA: An Analysis of How Language Models Perform with Context Permutation},
  author={Huang, Wenyu  and
      Vougiouklis, Pavlos  and
      Lapata, Mirella  and
      Pan, Jeff Z.},
  booktitle={Proceedings of ACL 2025},
  year={2025}
}
```