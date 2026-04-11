# plot_loss_compare.py
# loss曲线对比
# plot_loss_compare.py
# loss曲线对比 + 自动分析输出

import matplotlib.pyplot as plt
import numpy as np


def read_loss(path):
    train, val = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "train_loss" in line:
                parts = line.strip().split("|")
                train.append(float(parts[1].split("=")[1]))
                val.append(float(parts[2].split("=")[1]))
    return train, val


# 读取数据
exp1_train, exp1_val = read_loss("outputs/logs/exp1_log.txt")
exp2_train, exp2_val = read_loss("outputs/logs/exp2_log.txt")


# =========================
# 数值分析（新增）
# =========================
def analyze(name, train, val):
    best_epoch = int(np.argmin(val)) + 1
    best_val = float(np.min(val))
    final_val = float(val[-1])

    print(f"\n===== {name} =====")
    print(f"最佳epoch: {best_epoch}")
    print(f"最小val_loss: {best_val:.4f}")
    print(f"最终val_loss: {final_val:.4f}")
    print(f"是否过拟合: {'是' if final_val > best_val else '否'}")

    return best_epoch, best_val


e1_epoch, e1_best = analyze("exp1", exp1_train, exp1_val)
e2_epoch, e2_best = analyze("exp2", exp2_train, exp2_val)


# =========================
# 对比结论
# =========================
print("\n===== 对比分析 =====")

# 判断loss差异（设置阈值，避免误判）
diff = abs(e1_best - e2_best)
threshold = 0.01  # 小于这个认为“差不多”

if diff < threshold:
    print("exp1 与 exp2 验证损失接近（差异不显著）")
elif e2_best < e1_best:
    print("exp2 验证损失更低 → 泛化能力略强")
else:
    print("exp1 验证损失更低 → 泛化能力略强")


# 判断收敛速度
if e1_epoch == e2_epoch:
    print("两者收敛速度基本一致")
elif e2_epoch > e1_epoch:
    print("exp2 收敛更慢（需要更多epoch）")
else:
    print("exp1 收敛更慢")


# 判断过拟合程度
overfit1 = exp1_val[-1] - e1_best
overfit2 = exp2_val[-1] - e2_best

if overfit2 > overfit1:
    print("exp2 过拟合更明显")
elif overfit1 > overfit2:
    print("exp1 过拟合更明显")
else:
    print("两者过拟合程度相近")

# =========================
# 画图
# =========================
plt.figure(figsize=(8, 5))

plt.plot(exp1_val, label="exp1 val_loss")
plt.plot(exp2_val, label="exp2 val_loss")

# 标出最优点
plt.scatter(e1_epoch - 1, e1_best)
plt.scatter(e2_epoch - 1, e2_best)

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Validation Loss Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/figs/loss_compare.png", dpi=200)
plt.show()