import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch

SYSTEM_PROMPTS = {
    "answer_only_qwen": "You are Qwen and are now helping answer question based on some search result. Only provide the exact final answer and put it within \\boxed{}.",
    "answer_only_llama": "You are Llama and are now helping answer question based on some search result. Only provide the exact final answer and put it within \\boxed{}.",
    "cot": "Please reason step by step, and put your final answer within \\boxed{}.",
    "ft_qwen": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "ft_llama": "You are Llama, created by Meta. You are a helpful assistant."
}

PROMPTS = {
    "raw": "Answer the question using only the provided search results (some of which might be irrelevant). \n\n{}\n\nQuestion: {}\n.",
    "sup": "Answer the question using only the provided search results (some of which might be irrelevant). <instruct>\n\n{}\n\n<question>Question: {}\n",
}

def load_data(fpath):
    if fpath.endswith("jsonl"):
        data = []
        with open(fpath) as f:
            lines = f.read().splitlines()
            for line in lines:
                data.append(json.loads(line))
        return data
    return json.load(open(fpath))

def format_doc(ctx, index, with_special_token=False):
    if with_special_token:
        return "<doc>Document [{}](Title: {}) {}".format(index, ctx['title'], ctx['paragraph_text'])
    else:
        return "Document [{}](Title: {}) {}".format(index, ctx['title'], ctx['paragraph_text'])
    
def create_random_prompt(sample, doc_order, with_special_token=False):
    if with_special_token:
        prompt = PROMPTS["sup"]
    else:
        prompt = PROMPTS["raw"]
    doc_prompts = []
    idx = 0
    for doc_id in doc_order:
        doc_prompts.append(format_doc(sample["paragraphs"][doc_id], idx, with_special_token))
        idx += 1

    return prompt.format('\n'.join(doc_prompts), sample["question"])

def create_prompt(sample, with_special_token=False, order=None):
    if with_special_token:
        prompt = PROMPTS["sup"]
    else:
        prompt = PROMPTS["raw"]
            
    doc_orders = []
    for decomposition in sample["question_decomposition"]:
        doc_orders.append(int(decomposition["paragraph_support_idx"]))

    if not order:
        doc_ids = range(min(20, len(sample["paragraphs"])))
    elif order == "rm_first":
        doc_ids = [] 
        for i in range(min(20, len(sample["paragraphs"]))):
            if i == doc_orders[0]:
                continue
            else:
                doc_ids.append(i)
    elif order == "forward":
        doc_ids = []
        j = 0
        for i in range(len(sample["paragraphs"])):
            if i not in doc_orders:
                doc_ids.append(i)
            else:
                doc_ids.append(doc_orders[j])
                j += 1
    elif order == "backward":
        doc_ids = []
        j = len(doc_orders) - 1
        for i in range(len(sample["paragraphs"])):
            if i not in doc_orders:
                doc_ids.append(i)
            else:
                doc_ids.append(doc_orders[j])
                j -= 1
    elif order == "gold":
        doc_ids = []
        for i in range(len(sample["paragraphs"])):
            if i in doc_orders:
                doc_ids.append(i)
    elif order == "forward_0":
        doc_ids = []
        for i in range(len(sample["paragraphs"])):
            if i not in doc_orders:
                doc_ids.append(i)
        for doc_id in doc_orders:
            doc_ids.append(doc_id)
    elif order == "forward_1":
        doc_ids = []
        for i in range(len(sample["paragraphs"])):
            if i not in doc_orders:
                doc_ids.append(i)
        noise_len = len(doc_ids)
        doc_orders.reverse()
        for j, doc_id in enumerate(doc_orders):
            doc_ids.insert(noise_len-j, doc_id)
    elif order == "forward_2":
        doc_ids = []
        for i in range(len(sample["paragraphs"])):
            if i not in doc_orders:
                doc_ids.append(i)
        noise_len = len(doc_ids)
        doc_orders.reverse()
        for j, doc_id in enumerate(doc_orders):
            doc_ids.insert(noise_len-j*2, doc_id)
    elif order == "forward_3":
        doc_ids = []
        for i in range(len(sample["paragraphs"])):
            if i not in doc_orders:
                doc_ids.append(i)
        noise_len = len(doc_ids)
        doc_orders.reverse()
        for j, doc_id in enumerate(doc_orders):
            doc_ids.insert(noise_len-j*3, doc_id)
    elif order == "forward_4":
        doc_ids = []
        for i in range(len(sample["paragraphs"])):
            if i not in doc_orders:
                doc_ids.append(i)
        noise_len = len(doc_ids)
        doc_orders.reverse()
        for j, doc_id in enumerate(doc_orders):
            doc_ids.insert(noise_len-j*4, doc_id)
    elif order == "forward_5":
        doc_ids = []
        for i in range(len(sample["paragraphs"])):
            if i not in doc_orders:
                doc_ids.append(i)
        noise_len = len(doc_ids)
        doc_orders.reverse()
        for j, doc_id in enumerate(doc_orders):
            doc_ids.insert(noise_len-j*5, doc_id)
    else:
        raise NotImplementedError


    doc_prompts = []
    idx = 0
    for doc_id in doc_ids:
        doc_prompts.append(format_doc(sample["paragraphs"][doc_id], idx, with_special_token))
        idx += 1

    return prompt.format('\n'.join(doc_prompts), sample["question"])


def tokenize_for_eval(prompt, prompt_sup, tokenizer, tokenizer_sup, speical_ids=[151665, 151666, 151667], ):
    sup_ids = tokenizer_sup(prompt_sup)["input_ids"]
    inst_id = np.where(np.array(sup_ids) == speical_ids[0])[0]
    doc_ids = np.where(np.array(sup_ids) == speical_ids[1])[0]
    doc_ids = doc_ids - range(2, len(doc_ids) + 2)
    question_id = np.where(np.array(sup_ids) == speical_ids[2])[0] - len(doc_ids) - 2
    model_inputs = tokenizer(prompt, return_tensors="pt")
    return (inst_id, doc_ids, question_id), model_inputs
    
def grouping_attention_score(attention_scores, special_ids):
    n_layers = len(attention_scores[0])
    n_attention_heads = attention_scores[0][0].shape[1]
    n_input_tokens = attention_scores[0][0].shape[2]
    grouped_length = 1 + len(special_ids[1]) + len(attention_scores)
    
    grouped_scores = torch.zeros([n_layers, n_attention_heads, grouped_length, grouped_length], dtype=torch.float16, device='cuda')
    group_ids = [0] + list(special_ids[1]) + list(special_ids[2]) + list(range(n_input_tokens, n_input_tokens + len(attention_scores)))
    
    for layer in range(n_layers):
        base_attention = torch.tensor(attention_scores[0][layer][0], dtype=torch.float16, device='cuda')

        for i in range(grouped_length):
            for j in range(i + 1):
                if i < len(group_ids) - len(attention_scores):
                    start_i, end_i = group_ids[i], group_ids[i + 1]
                    start_j, end_j = group_ids[j], group_ids[j + 1]
                    grouped_scores[layer, :, i, j] = torch.sum(base_attention[:, start_i:end_i, start_j:end_j], dim=(1, 2)) / (end_i - start_i)
                else:
                    token_idx = i - len(group_ids) + len(attention_scores) + 1
                    start_j, end_j = group_ids[j], group_ids[j + 1]
                    grouped_scores[layer, :, i, j] = torch.sum(attention_scores[token_idx][layer][0, :, 0, start_j:end_j], dim=1)
    
    return grouped_scores.cpu().numpy(), group_ids


def compute_influence(grouped_scores, num_docs=20, answer_tokens=None, accumulate=False, num_max_docs=20):
    n_layers = grouped_scores.shape[0]
    n_attention_heads = grouped_scores.shape[1]
    n_group_length = grouped_scores.shape[2]
    n_generated_tokens = n_group_length - num_docs - 2
    generate_start = n_group_length - n_generated_tokens
    if not answer_tokens:
        answer_tokens = [0, n_generated_tokens]
    if answer_tokens:
        generate_start = generate_start + answer_tokens[0]
    
    ics_pl_f = np.zeros([n_layers, n_attention_heads, num_docs])
    ics_pl_a = np.zeros([n_layers, n_attention_heads, num_docs])
    for layer in range(n_layers):
        for head in range(n_attention_heads):
            if accumulate and layer > 0:
                ics_pl_f[layer][head] = ics_pl_f[layer-1][head] + grouped_scores[layer][head][generate_start][1:1+num_docs]
                if answer_tokens:
                    ics_pl_a[layer][head] = ics_pl_a[layer-1][head] + np.average(grouped_scores[layer][head][generate_start:generate_start+answer_tokens[1],1:1+num_docs], axis=0)
                else:
                    ics_pl_a[layer][head] = ics_pl_a[layer-1][head] + np.average(grouped_scores[layer][head][generate_start:,1:1+num_docs], axis=0)


            else:
                ics_pl_f[layer][head] = grouped_scores[layer][head][generate_start][1:1+num_docs]
                if answer_tokens:
                    ics_pl_a[layer][head] = np.average(grouped_scores[layer][head][generate_start:generate_start+answer_tokens[1],1:1+num_docs], axis=0)
                else:
                    ics_pl_a[layer][head] = np.average(grouped_scores[layer][head][generate_start:,1:1+num_docs], axis=0)
                 

    if num_docs < num_max_docs:
        n_ics_pl_a = np.zeros([n_layers, n_attention_heads, num_max_docs])
        n_ics_pl_f = np.zeros([n_layers, n_attention_heads, num_max_docs])
        n_ics_pl_a[:, :, :ics_pl_a.shape[2]] = ics_pl_a
        n_ics_pl_f[:, :, :ics_pl_f.shape[2]] = ics_pl_f
        return n_ics_pl_f, n_ics_pl_a
    return ics_pl_f, ics_pl_a
                