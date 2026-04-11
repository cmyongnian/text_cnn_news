# visualize_compare.py
# 指标对比柱状图
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/reports/results_comparison.csv")

metrics = ["test_acc", "test_macro_f1"]

exp1 = df[df["experiment"] == "exp1"].iloc[0]
exp2 = df[df["experiment"] == "exp2"].iloc[0]

labels = ["Accuracy", "Macro-F1"]

exp1_vals = [exp1["test_acc"], exp1["test_macro_f1"]]
exp2_vals = [exp2["test_acc"], exp2["test_macro_f1"]]

x = range(len(labels))

plt.figure(figsize=(7,5))
plt.bar([i - 0.2 for i in x], exp1_vals, width=0.4, label="exp1")
plt.bar([i + 0.2 for i in x], exp2_vals, width=0.4, label="exp2")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Exp1 vs Exp2 Performance Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/figs/exp_compare_bar.png", dpi=200)
plt.show()