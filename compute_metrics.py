from argparse import ArgumentParser
import json
import numpy as np
import re

def load_data(fpath):
    if fpath.endswith("jsonl"):
        data = []
        with open(fpath) as f:
            lines = f.read().splitlines()
            for line in lines:
                data.append(json.loads(line))
        return data
    return json.load(open(fpath))


def compute_em(ref_sample, prediction):
    # Function to remove special characters and normalize text
    def normalize(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

    if "answer_aliases" in ref_sample:
        ref_answers = [ref_sample["answer"]] + ref_sample["answer_aliases"]
    else:
        ref_answers = [ref_sample["answer"]]
    
    normalized_prediction = normalize(prediction)
    
    for ref_answer in ref_answers:
        normalized_ref_answer = normalize(ref_answer)
        if normalized_ref_answer in normalized_prediction:
            return 1.0
    return 0.0

def get_cot_ans(pred):
    pattern = r'\\boxed{([^}]+)}'

    # Find all matches
    matches = re.findall(pattern, pred)
    if matches:
        return matches[0]

    return ""

def compute_metrics(args):
    reference = load_data(args.ref_path)
    predictions = json.load(open(args.pred_path, "r"))["predictions"]
    ems = []

    for ref_sample, pred_sample in zip(reference, predictions):
        prediction = pred_sample["prediction"]
        if args.cot:
            prediction = get_cot_ans(prediction)
        else:
            prediction = prediction.strip().split("\n")[-1]
        em = compute_em(ref_sample, prediction)
        ems.append(em)


    print(f"EM: {np.mean(ems)}")

    
def main():
    parser = ArgumentParser(description="Example script using ArgumentParser")
    parser.add_argument('--ref_path', type=str, default="data/musique_ans_v1.0_dev.jsonl", help="Path to the reference file")
    parser.add_argument('--pred_path', type=str, required=True, help="Path to the prediction file")
    parser.add_argument('--cot', action="store_true", help="Use \\boxed{} to extract the answer from the prediction")

    args = parser.parse_args()
    compute_metrics(args)

if __name__ == "__main__":
    main()
