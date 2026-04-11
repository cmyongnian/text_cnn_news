# run_error_compare.py

import json
from utils import read_jsonl
from compare_errors import compare_error_analysis, plot_compare


test_raw = read_jsonl("data/raw/test.json")

pred1 = json.load(open("outputs/preds_exp1.json"))
pred2 = json.load(open("outputs/preds_exp2.json"))

improved, degraded, both_wrong, both_right = compare_error_analysis(
    test_raw, pred1, pred2
)

print("Improved:", improved)
print("Degraded:", degraded)
print("Both Wrong:", both_wrong)
print("Both Right:", both_right)

plot_compare(improved, degraded, both_wrong, both_right)