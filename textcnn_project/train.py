from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model_textcnn import TextCNN
from model_textcnn_fusion import KeywordFusionTextCNN
from utils import (
    PAD_TOKEN,
    build_model_summary,
    ensure_dir,
    evaluate,
    load_json,
    load_label_mapping,
    load_yaml,
    plot_loss_curves,
    read_jsonl,
    save_text,
    set_seed,
    update_results_csv,
    upsert_section,
)


class NewsDataset(Dataset):
    def __init__(self, seqs, labels, key_masks=None):
        self.seqs = seqs
        self.labels = labels
        self.key_masks = key_masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.seqs[idx],
            "label": self.labels[idx],
        }
        if self.key_masks is not None:
            item["key_mask"] = self.key_masks[idx]
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="训练 TextCNN / 关键词融合 TextCNN")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--exp", type=str, choices=["exp1", "exp2"], required=True)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, use_keywords: bool):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_keywords:
            key_mask = batch["key_mask"].to(device)
            logits = model(input_ids, key_mask)
        else:
            logits = model(input_ids)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_yaml(project_root / args.config)

    set_seed(config["seed"])

    raw_dir = project_root / config["paths"]["raw_dir"]
    processed_dir = project_root / config["paths"]["processed_dir"]
    output_dir = project_root / config["paths"]["output_dir"]

    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    reports_dir = output_dir / "reports"
    figs_dir = output_dir / "figs"

    for d in [models_dir, logs_dir, reports_dir, figs_dir]:
        ensure_dir(d)

    vocab_path = processed_dir / "vocab.json"
    train_seq_path = processed_dir / "train_seq.pt"
    test_seq_path = processed_dir / "test_seq.pt"
    train_key_mask_path = processed_dir / "train_key_mask.pt"
    test_key_mask_path = processed_dir / "test_key_mask.pt"

    required_files = [vocab_path, train_seq_path, test_seq_path, train_key_mask_path, test_key_mask_path]
    missing_files = [str(p) for p in required_files if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            "缺少预处理文件，请先运行：python preprocess.py\n缺少文件：\n" + "\n".join(missing_files)
        )

    vocab_obj = load_json(vocab_path)
    token2id = vocab_obj["token2id"] if "token2id" in vocab_obj else vocab_obj

    train_seq = torch.load(train_seq_path)
    test_seq = torch.load(test_seq_path)
    train_key_mask = torch.load(train_key_mask_path)
    test_key_mask = torch.load(test_key_mask_path)

    label2id, id2label, id2desc = load_label_mapping(raw_dir / "labels.json")
    train_raw = read_jsonl(raw_dir / "train.json")
    test_raw = read_jsonl(raw_dir / "test.json")

    train_labels = torch.tensor([label2id[str(x["label"])] for x in train_raw], dtype=torch.long)
    test_labels = torch.tensor([label2id[str(x["label"])] for x in test_raw], dtype=torch.long)

    train_idx, val_idx = train_test_split(
        np.arange(len(train_labels)),
        test_size=config["train"]["val_ratio"],
        random_state=config["seed"],
        stratify=train_labels.numpy(),
    )

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)

    use_keywords = args.exp == "exp2"

    train_dataset = NewsDataset(
        seqs=train_seq[train_idx],
        labels=train_labels[train_idx],
        key_masks=train_key_mask[train_idx] if use_keywords else None,
    )
    val_dataset = NewsDataset(
        seqs=train_seq[val_idx],
        labels=train_labels[val_idx],
        key_masks=train_key_mask[val_idx] if use_keywords else None,
    )
    test_dataset = NewsDataset(
        seqs=test_seq,
        labels=test_labels,
        key_masks=test_key_mask if use_keywords else None,
    )

    batch_size = config["train"]["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = len(token2id)
    num_classes = config["num_classes"]
    pad_idx = token2id[PAD_TOKEN]

    if use_keywords:
        model = KeywordFusionTextCNN(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=config["model"]["embed_dim"],
            num_filters=config["model"]["num_filters"],
            kernel_sizes=tuple(config["model"]["kernel_sizes"]),
            dropout=config["model"]["dropout"],
            pad_idx=pad_idx,
            keyword_scale=config["model"]["keyword_scale"],
        )
    else:
        model = TextCNN(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=config["model"]["embed_dim"],
            num_filters=config["model"]["num_filters"],
            kernel_sizes=tuple(config["model"]["kernel_sizes"]),
            dropout=config["model"]["dropout"],
            pad_idx=pad_idx,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    epochs = config["train"]["epochs"]
    patience = config["train"]["patience"]

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_count = 0

    train_losses = []
    val_losses = []
    log_lines = []

    model_path = models_dir / f"best_model_{args.exp}.pth"
    log_path = logs_dir / f"{args.exp}_log.txt"
    fig_path = figs_dir / f"{args.exp}_loss_curve.png"
    report_path = reports_dir / f"classification_report_{args.exp}.txt"
    summary_path = reports_dir / "model_summary.txt"
    results_csv_path = reports_dir / "results_comparison.csv"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_keywords=use_keywords,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_keywords=use_keywords,
            id2desc=id2desc,
        )

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])

        line = (
            f"epoch={epoch:02d} | "
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
                stop_msg = f"early stopping triggered at epoch {epoch}"
                print(stop_msg)
                log_lines.append(stop_msg)
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

    report_text = "\n".join(
        [
            f"experiment: {args.exp}",
            f"use_keywords: {use_keywords}",
            f"best_epoch: {best_epoch}",
            f"best_val_macro_f1: {best_val_macro_f1:.4f}",
            f"test_loss: {test_metrics['loss']:.4f}",
            f"test_acc: {test_metrics['acc']:.4f}",
            f"test_macro_f1: {test_metrics['macro_f1']:.4f}",
            "",
            "classification_report",
            "=" * 60,
            test_metrics["report"],
        ]
    )
    save_text(report_path, report_text)

    plot_loss_curves(train_losses, val_losses, fig_path)

    summary_text = build_model_summary(model)
    upsert_section(summary_path, f"{args.exp}_model_summary", summary_text)

    update_results_csv(
        results_csv_path,
        {
            "experiment": args.exp,
            "use_keywords": use_keywords,
            "best_epoch": best_epoch,
            "best_val_macro_f1": f"{best_val_macro_f1:.4f}",
            "test_loss": f"{test_metrics['loss']:.4f}",
            "test_acc": f"{test_metrics['acc']:.4f}",
            "test_macro_f1": f"{test_metrics['macro_f1']:.4f}",
            "model_path": str(model_path),
            "report_path": str(report_path),
            "figure_path": str(fig_path),
        },
    )

    tail_lines = [
        "",
        f"best_epoch={best_epoch}",
        f"best_val_macro_f1={best_val_macro_f1:.4f}",
        f"test_loss={test_metrics['loss']:.4f}",
        f"test_acc={test_metrics['acc']:.4f}",
        f"test_macro_f1={test_metrics['macro_f1']:.4f}",
        f"elapsed_seconds={elapsed:.2f}",
        f"model_path={model_path}",
        f"report_path={report_path}",
        f"figure_path={fig_path}",
    ]
    log_lines.extend(tail_lines)
    save_text(log_path, "\n".join(log_lines))

    print("\n训练完成。")
    print(f"最佳模型：{model_path}")
    print(f"训练日志：{log_path}")
    print(f"分类报告：{report_path}")
    print(f"损失曲线：{fig_path}")
    print(f"结果对比表：{results_csv_path}")


if __name__ == "__main__":
    main()