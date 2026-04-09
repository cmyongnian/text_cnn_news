from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path, obj: Any):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path, text: str):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_label_mapping(label_path):
    """
    labels.json:
    {"label": "100", "label_desc": "news_story"}
    ...
    映射为连续类别 id，供 CrossEntropyLoss 使用。
    """
    label2id = {}
    id2label = {}
    id2desc = {}

    with open(label_path, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = str(obj["label"])
            label_desc = obj.get("label_desc", label)

            label2id[label] = idx
            id2label[idx] = label
            id2desc[idx] = label_desc
            idx += 1

    return label2id, id2label, id2desc


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device, use_keywords: bool, id2desc: dict[int, str]):
    """
    通用评估函数：
    - exp1: 只输入 input_ids
    - exp2: 输入 input_ids + key_mask
    返回 loss / acc / macro_f1 / classification_report
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            if use_keywords:
                key_mask = batch["key_mask"].to(device)
                logits = model(input_ids, key_mask)
            else:
                logits = model(input_ids)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    label_ids = list(range(len(id2desc)))
    target_names = [id2desc[i] for i in label_ids]

    report = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "report": report,
        "preds": all_preds,
        "labels": all_labels,
    }


def plot_loss_curves(train_losses, val_losses, save_path):
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def build_model_summary(model) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        repr(model),
        "",
        f"total_params: {total_params}",
        f"trainable_params: {trainable_params}",
        "",
        "parameter_shapes:",
    ]

    for name, param in model.named_parameters():
        lines.append(
            f"- {name}: shape={list(param.shape)}, numel={param.numel()}, trainable={param.requires_grad}"
        )

    return "\n".join(lines)


def upsert_section(path, section_name: str, content: str):
    """
    把一段内容按 section_name 更新到同一个文件中。
    适合把 exp1 / exp2 的模型摘要写进同一个 model_summary.txt。
    """
    path = Path(path)
    ensure_dir(path.parent)

    start = f"===== {section_name} START ====="
    end = f"===== {section_name} END ====="
    block = f"{start}\n{content.rstrip()}\n{end}\n"

    if not path.exists():
        save_text(path, block)
        return

    original = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\n?", re.S)

    if pattern.search(original):
        updated = pattern.sub(block, original)
    else:
        updated = original.rstrip() + "\n\n" + block

    save_text(path, updated)


def update_results_csv(csv_path, row: dict):
    """
    更新/写入结果对比表 results_comparison.csv
    如果 experiment 已存在，则覆盖该行；否则追加。
    """
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    fieldnames = [
        "experiment",
        "use_keywords",
        "best_epoch",
        "best_val_macro_f1",
        "test_loss",
        "test_acc",
        "test_macro_f1",
        "model_path",
        "report_path",
        "figure_path",
    ]

    rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    experiment_name = str(row["experiment"])
    replaced = False

    for i, existing in enumerate(rows):
        if existing.get("experiment") == experiment_name:
            rows[i] = {k: str(row.get(k, "")) for k in fieldnames}
            replaced = True
            break

    if not replaced:
        rows.append({k: str(row.get(k, "")) for k in fieldnames})

    rows.sort(key=lambda x: x["experiment"])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)