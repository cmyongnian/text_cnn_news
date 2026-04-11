# compare_errors.py
# 
import json
import matplotlib.pyplot as plt
from utils import read_jsonl
from utils import load_label_mapping

def compare_error_analysis(test_raw, pred1, pred2):
    
    label2id, _, _ = load_label_mapping("data/raw/labels.json")

    improved = 0
    degraded = 0
    both_wrong = 0
    both_right = 0

    for i in range(len(pred1)):
        gold = label2id[str(test_raw[i]["label"])]

        p1 = pred1[i]
        p2 = pred2[i]

        if p1 != gold and p2 == gold:
            improved += 1
        elif p1 == gold and p2 != gold:
            degraded += 1
        elif p1 != gold and p2 != gold:
            both_wrong += 1
        else:
            both_right += 1

    return improved, degraded, both_wrong, both_right

def plot_compare(improved, degraded, both_wrong, both_right):
    labels = ["Improved", "Degraded", "Both Wrong", "Both Right"]
    values = [improved, degraded, both_wrong, both_right]

    plt.figure(figsize=(7,5))
    plt.bar(labels, values)
    plt.title("Exp1 vs Exp2 Error Comparison")
    plt.ylabel("Sample Count")

    for i, v in enumerate(values):
        plt.text(i, v + 5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig("outputs/figs/error_compare.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    test_raw = read_jsonl("data/raw/test.json")

    print("需要接入pred1/pred2")