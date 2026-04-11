import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model_textcnn import TextCNN
from model_textcnn_fusion import DualBranchTextCNN


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return items

    if text.startswith("["):
        return json.loads(text)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def resolve_raw_dir(config):
    raw_dir = Path(config["paths"]["raw_dir"])
    need_files = ["labels.json", "train.json", "test.json"]

    if all((raw_dir / x).exists() for x in need_files):
        return raw_dir

    fallback = Path("data")
    if all((fallback / x).exists() for x in need_files):
        return fallback

    raise FileNotFoundError(
        "找不到原始数据。请把 labels.json、train.json、test.json 放到 data/raw/ 或 data/ 目录。"
    )


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NewsDataset(Dataset):
    def __init__(self, seqs, labels, keyword_seqs=None):
        self.seqs = seqs
        self.labels = labels
        self.keyword_seqs = keyword_seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.seqs[idx],
            "label": self.labels[idx],
        }
        if self.keyword_seqs is not None:
            item["keyword_ids"] = self.keyword_seqs[idx]
        return item


def load_label_mapping(labels_path: Path):
    labels_raw = read_jsonl(labels_path)
    label2id = {}
    id2label = {}
    id2desc = {}

    for idx, item in enumerate(labels_raw):
        raw_label = str(item["label"])
        desc = item.get("label_desc", raw_label)
        label2id[raw_label] = idx
        id2label[idx] = raw_label
        id2desc[idx] = desc

    return label2id, id2label, id2desc


def build_class_weights(labels: torch.Tensor, num_classes: int):
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights


def train_one_epoch(model, loader, criterion, optimizer, device, use_keywords: bool, grad_clip: float):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_keywords:
            keyword_ids = batch["keyword_ids"].to(device)
            logits = model(input_ids, keyword_ids)
        else:
            logits = model(input_ids)

        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_keywords: bool, id2desc):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        if use_keywords:
            keyword_ids = batch["keyword_ids"].to(device)
            logits = model(input_ids, keyword_ids)
        else:
            logits = model(input_ids)

        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    target_names = [id2desc[i] for i in range(len(id2desc))]
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report_text = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(target_names))),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(target_names))))

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "report_text": report_text,
        "report_dict": report_dict,
        "cm": cm,
        "labels": all_labels,
        "preds": all_preds,
    }


def plot_training_curves(train_losses, val_losses, val_f1s, save_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.plot(val_f1s, label="验证集 Macro-F1")
    plt.xlabel("训练轮次")
    plt.ylabel("数值")
    plt.title("训练过程曲线")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, id2desc, save_path: Path, title: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.colorbar()
    names = [id2desc[i] for i in range(len(id2desc))]
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(report_dict, id2desc, save_path: Path, title: str):
    names = [id2desc[i] for i in range(len(id2desc))]
    precision = [report_dict.get(name, {}).get("precision", 0.0) for name in names]
    recall = [report_dict.get(name, {}).get("recall", 0.0) for name in names]
    f1s = [report_dict.get(name, {}).get("f1-score", 0.0) for name in names]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precision, width=width, label="准确率 Precision")
    plt.bar(x, recall, width=width, label="召回率 Recall")
    plt.bar(x + width, f1s, width=width, label="F1 值")
    plt.xticks(x, names, rotation=75)
    plt.ylim(0, 1.05)
    plt.ylabel("分数")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_prediction_details(raw_items, true_ids, pred_ids, id2desc, save_path: Path):
    rows = []
    for item, y_true, y_pred in zip(raw_items, true_ids, pred_ids):
        rows.append(
            {
                "sentence": item.get("sentence", ""),
                "keywords": item.get("keywords", ""),
                "true_label_desc": id2desc[y_true],
                "pred_label_desc": id2desc[y_pred],
                "is_correct": bool(y_true == y_pred),
                "sentence_char_len": len(str(item.get("sentence", "") or "")),
                "keywords_char_len": len(str(item.get("keywords", "") or "")),
            }
        )
    save_json(rows, save_path)


def save_badcases(raw_items, true_ids, pred_ids, id2desc, save_path: Path):
    rows = []
    for item, y_true, y_pred in zip(raw_items, true_ids, pred_ids):
        if y_true != y_pred:
            rows.append(
                {
                    "sentence": item.get("sentence", ""),
                    "keywords": item.get("keywords", ""),
                    "true_label_desc": id2desc[y_true],
                    "pred_label_desc": id2desc[y_pred],
                    "sentence_char_len": len(str(item.get("sentence", "") or "")),
                    "keywords_char_len": len(str(item.get("keywords", "") or "")),
                }
            )
    save_json(rows, save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["exp1", "exp2"])
    return parser.parse_args()


def main():
    setup_chinese_font()
    args = parse_args()

    config = load_yaml(Path("config.yaml"))
    set_seed(int(config["seed"]))

    raw_dir = resolve_raw_dir(config)
    processed_dir = Path(config["paths"]["processed_dir"])
    output_dir = Path(config["paths"]["output_dir"])

    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    figs_dir = output_dir / "figs"
    logs_dir = output_dir / "logs"

    for d in [models_dir, reports_dir, figs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    vocab_path = processed_dir / "vocab.json"
    train_seq_path = processed_dir / "train_seq.pt"
    test_seq_path = processed_dir / "test_seq.pt"
    train_kw_seq_path = processed_dir / "train_kw_seq.pt"
    test_kw_seq_path = processed_dir / "test_kw_seq.pt"

    required = [vocab_path, train_seq_path, test_seq_path, train_kw_seq_path, test_kw_seq_path]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("缺少预处理文件，请先运行 python preprocess.py\n" + "\n".join(missing))

    vocab_obj = load_json(vocab_path)
    token2id = vocab_obj["token2id"]
    pad_idx = token2id[vocab_obj.get("pad_token", "[PAD]")]
    vocab_size = len(token2id)

    train_seq = torch.load(train_seq_path)
    test_seq = torch.load(test_seq_path)
    train_kw_seq = torch.load(train_kw_seq_path)
    test_kw_seq = torch.load(test_kw_seq_path)

    label2id, _, id2desc = load_label_mapping(raw_dir / "labels.json")
    train_raw = read_jsonl(raw_dir / "train.json")
    test_raw = read_jsonl(raw_dir / "test.json")

    train_labels = torch.tensor([label2id[str(x["label"])] for x in train_raw], dtype=torch.long)
    test_labels = torch.tensor([label2id[str(x["label"])] for x in test_raw], dtype=torch.long)

    indices = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(config["train"]["val_ratio"]),
        random_state=int(config["seed"]),
        stratify=train_labels.numpy(),
    )

    use_keywords = args.exp == "exp2"

    train_dataset = NewsDataset(
        seqs=train_seq[train_idx],
        labels=train_labels[train_idx],
        keyword_seqs=train_kw_seq[train_idx] if use_keywords else None,
    )
    val_dataset = NewsDataset(
        seqs=train_seq[val_idx],
        labels=train_labels[val_idx],
        keyword_seqs=train_kw_seq[val_idx] if use_keywords else None,
    )
    test_dataset = NewsDataset(
        seqs=test_seq,
        labels=test_labels,
        keyword_seqs=test_kw_seq if use_keywords else None,
    )

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_kwargs = dict(
        vocab_size=vocab_size,
        num_classes=int(config["num_classes"]),
        embed_dim=int(config["model"]["embed_dim"]),
        num_filters=int(config["model"]["num_filters"]),
        kernel_sizes=tuple(config["model"]["kernel_sizes"]),
        dropout=float(config["model"]["dropout"]),
        emb_dropout=float(config["model"].get("emb_dropout", 0.1)),
        pad_idx=pad_idx,
    )

    if use_keywords:
        model = DualBranchTextCNN(
            **model_kwargs,
            fusion_hidden_dim=int(config["model"].get("fusion_hidden_dim", 128)),
        )
    else:
        model = TextCNN(**model_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    class_weights = build_class_weights(train_labels[train_idx], int(config["num_classes"]))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["train"].get("lr_factor", 0.5)),
        patience=int(config["train"].get("lr_patience", 1)),
    )

    epochs = int(config["train"]["epochs"])
    patience = int(config["train"]["patience"])
    grad_clip = float(config["train"].get("grad_clip", 0.0))

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    train_losses = []
    val_losses = []
    val_f1s = []
    log_lines = []

    model_path = models_dir / f"best_model_{args.exp}.pth"
    curve_path = figs_dir / f"{args.exp}_训练曲线.png"
    cm_path = figs_dir / f"{args.exp}_混淆矩阵.png"
    per_class_path = figs_dir / f"{args.exp}_各类别指标.png"
    log_path = logs_dir / f"{args.exp}_train_log.txt"
    report_path = reports_dir / f"classification_report_{args.exp}.txt"
    metrics_path = reports_dir / f"metrics_{args.exp}.json"
    pred_path = reports_dir / f"predictions_{args.exp}.json"
    badcase_path = reports_dir / f"badcases_{args.exp}.json"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_keywords=use_keywords,
            grad_clip=grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_keywords=use_keywords,
            id2desc=id2desc,
        )

        scheduler.step(val_metrics["macro_f1"])

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        val_f1s.append(val_metrics["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]
        line = (
            f"epoch={epoch:02d} | "
            f"lr={current_lr:.8f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        print(line)
        log_lines.append(line)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                log_lines.append(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_keywords=use_keywords,
        id2desc=id2desc,
    )

    elapsed = time.time() - start_time

    plot_training_curves(train_losses, val_losses, val_f1s, curve_path)
    plot_confusion(
        cm=test_metrics["cm"],
        id2desc=id2desc,
        save_path=cm_path,
        title=f"{args.exp} 测试集混淆矩阵",
    )
    plot_per_class_metrics(
        report_dict=test_metrics["report_dict"],
        id2desc=id2desc,
        save_path=per_class_path,
        title=f"{args.exp} 测试集各类别指标",
    )

    save_text("\n".join(log_lines), log_path)

    report_text = "\n".join(
        [
            f"experiment: {args.exp}",
            f"use_keywords: {use_keywords}",
            f"best_epoch: {best_epoch}",
            f"best_val_macro_f1: {best_val_macro_f1:.4f}",
            f"test_loss: {test_metrics['loss']:.4f}",
            f"test_acc: {test_metrics['acc']:.4f}",
            f"test_macro_f1: {test_metrics['macro_f1']:.4f}",
            f"elapsed_seconds: {elapsed:.2f}",
            "",
            "classification_report",
            "=" * 60,
            test_metrics["report_text"],
        ]
    )
    save_text(report_text, report_path)

    metrics_obj = {
        "experiment": args.exp,
        "use_keywords": use_keywords,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(float(best_val_macro_f1), 6),
        "test_loss": round(float(test_metrics["loss"]), 6),
        "test_acc": round(float(test_metrics["acc"]), 6),
        "test_macro_f1": round(float(test_metrics["macro_f1"]), 6),
        "elapsed_seconds": round(float(elapsed), 2),
    }
    save_json(metrics_obj, metrics_path)

    save_prediction_details(
        raw_items=test_raw,
        true_ids=test_metrics["labels"],
        pred_ids=test_metrics["preds"],
        id2desc=id2desc,
        save_path=pred_path,
    )

    save_badcases(
        raw_items=test_raw,
        true_ids=test_metrics["labels"],
        pred_ids=test_metrics["preds"],
        id2desc=id2desc,
        save_path=badcase_path,
    )

    print("=" * 60)
    print(json.dumps(metrics_obj, ensure_ascii=False, indent=2))
    print(f"模型已保存到: {model_path}")
    print(f"训练曲线已保存到: {curve_path}")
    print(f"混淆矩阵已保存到: {cm_path}")
    print(f"各类别指标图已保存到: {per_class_path}")
    print(f"分类报告已保存到: {report_path}")
    print(f"预测明细已保存到: {pred_path}")
    print(f"错分样本已保存到: {badcase_path}")


if __name__ == "__main__":
    main()