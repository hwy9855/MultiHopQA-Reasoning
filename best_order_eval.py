from utils import load_data
from argparse import ArgumentParser
from compute_metrics import compute_em
import numpy as np

def eval_best_order(args):
    ref_data = load_data(args.ref_path)
    pred_data = load_data(args.pred_path)
    id2data = {}

    for sample in ref_data:
        id2data[sample["id"]] = sample

    best_order_ems = []
    for pred_sample in pred_data:
        if "id" not in pred_sample["best_sample"]:
            continue
        id = pred_sample["best_sample"]["id"] 
        if id in id2data:
            ref_sample = id2data[id]
            if pred_sample["all_samples"]:
                em = compute_em(ref_sample, pred_sample["best_sample"]["prediction"])
            else:
                em = 0
            if em:
                best_order_ems.append(1)
            else:
                best_order_ems.append(0)
    print(f"Best order EM: {np.mean(best_order_ems)}")

    average_ems = []
    for pred_sample in pred_data:
        if "id" not in pred_sample["best_sample"]:
            continue
        id = pred_sample["best_sample"]["id"] 
        if id in id2data:
            ref_sample = id2data[id]
            for s in range(len(pred_sample["all_samples"])):
                if pred_sample["all_samples"]:
                    em = compute_em(ref_sample, pred_sample["all_samples"][s]["prediction"])
                else:
                    em = 0
                if em:
                    average_ems.append(1)
                else:
                    average_ems.append(0)
    print(f"Average EM: {np.mean(average_ems)}")

def main():
    parser = ArgumentParser(description="Example script using ArgumentParser")
    parser.add_argument('--ref_path', type=str, help='Path to the reference file')
    parser.add_argument('--pred_path', type=str, help='Path to the prediction file (best order experiment)')
    args = parser.parse_args()
    eval_best_order(args)

if __name__ == "__main__":
    main()
