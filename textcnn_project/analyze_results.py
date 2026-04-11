import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def normalize_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\d+(\.\d+)?", " NUM ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_tokens(tokens):
    return [tok.strip() for tok in tokens if tok and tok.strip()]


def tokenize_sentence(text: str):
    return clean_tokens(jieba.lcut(normalize_text(text), cut_all=False))


def tokenize_keywords(keywords: str):
    keywords = normalize_text(keywords)
    if not keywords:
        return []
    parts = [p.strip() for p in re.split(r"[，,；;、\s]+", keywords) if p.strip()]
    tokens = []
    for part in parts:
        tokens.extend(clean_tokens(jieba.lcut(part, cut_all=False)))
    return tokens


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
    id2desc = {}
    for idx, item in enumerate(labels_raw):
        raw_label = str(item["label"])
        desc = item.get("label_desc", raw_label)
        label2id[raw_label] = idx
        id2desc[idx] = desc
    return label2id, id2desc


def build_model(exp_name, config, vocab_size, pad_idx):
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

    if exp_name == "exp2":
        return DualBranchTextCNN(
            **model_kwargs,
            fusion_hidden_dim=int(config["model"].get("fusion_hidden_dim", 128)),
        )
    return TextCNN(**model_kwargs)


@torch.no_grad()
def predict(model, loader, device, use_keywords: bool):
    model.eval()
    all_labels = []
    all_preds = []
    all_confs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        if use_keywords:
            keyword_ids = batch["keyword_ids"].to(device)
            logits = model(input_ids, keyword_ids)
        else:
            logits = model(input_ids)

        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

    return all_labels, all_preds, all_confs


def get_sample_features(item, max_len):
    sentence = item.get("sentence", "")
    keywords = item.get("keywords", "")

    sent_tokens = tokenize_sentence(sentence)
    kw_tokens = tokenize_keywords(keywords)

    sent_token_set = set(sent_tokens)
    kw_token_set = set(kw_tokens)

    overlap_count = sum(1 for x in kw_token_set if x in sent_token_set)
    overlap_ratio = overlap_count / max(1, len(kw_token_set))

    return {
        "正文": sentence,
        "关键词": keywords,
        "正文字符长度": len(str(sentence)),
        "关键词字符长度": len(str(keywords)),
        "正文分词长度": len(sent_tokens),
        "关键词分词长度": len(kw_tokens),
        "关键词重合词数": overlap_count,
        "关键词重合率": round(overlap_ratio, 4),
        "存在截断风险": bool(len(sent_tokens) > max_len),
        "正文过短": bool(len(sent_tokens) <= 8),
        "关键词缺失或极少": bool(len(kw_tokens) <= 1),
        "关键词重合度低": bool(len(kw_tokens) > 0 and overlap_ratio < 0.34),
    }


def plot_bar(values_a, values_b, labels, legend_a, legend_b, title, ylabel, save_path, rotation=75):
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, values_a, width=width, label=legend_a)
    plt.bar(x + width / 2, values_b, width=width, label=legend_b)
    plt.xticks(x, labels, rotation=rotation)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_hist(values_train, values_test, bins, title, xlabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(values_train, bins=bins, alpha=0.6, label="训练集")
    plt.hist(values_test, bins=bins, alpha=0.6, label="测试集")
    plt.xlabel(xlabel)
    plt.ylabel("样本数")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, id2desc, save_path, title):
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


def plot_per_class_metrics(report_dict, id2desc, save_path, title):
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


def bucket_name(length: int):
    if length <= 20:
        return "0-20"
    elif length <= 40:
        return "21-40"
    elif length <= 60:
        return "41-60"
    elif length <= 80:
        return "61-80"
    elif length <= 120:
        return "81-120"
    else:
        return "120+"


def plot_length_bucket_accuracy(raw_items, true_ids, pred_ids, field_name, save_path, title):
    bucket_total = {}
    bucket_correct = {}

    for item, y_true, y_pred in zip(raw_items, true_ids, pred_ids):
        text = str(item.get(field_name, "") or "")
        length = len(text)
        b = bucket_name(length)
        bucket_total[b] = bucket_total.get(b, 0) + 1
        bucket_correct[b] = bucket_correct.get(b, 0) + int(y_true == y_pred)

    ordered = ["0-20", "21-40", "41-60", "61-80", "81-120", "120+"]
    xs = []
    accs = []
    counts = []
    for b in ordered:
        if bucket_total.get(b, 0) > 0:
            xs.append(b)
            counts.append(bucket_total[b])
            accs.append(bucket_correct.get(b, 0) / bucket_total[b])

    plt.figure(figsize=(8, 5))
    bars = plt.bar(xs, accs)
    plt.ylim(0, 1.05)
    plt.xlabel("长度分桶")
    plt.ylabel("准确率")
    plt.title(title)

    for bar, acc, cnt in zip(bars, accs, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(acc + 0.02, 1.02),
            f"{acc:.2f}\n(n={cnt})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_top_confusions(cm, id2desc, save_path, topn=10):
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), f"{id2desc[i]} -> {id2desc[j]}"))

    pairs.sort(key=lambda x: x[0], reverse=True)
    top_pairs = pairs[:topn]

    if not top_pairs:
        return []

    counts = [x[0] for x in top_pairs][::-1]
    labels = [x[1] for x in top_pairs][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, counts)
    plt.xlabel("错分次数")
    plt.title("最常见错分类别对")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return top_pairs


def plot_reason_counts(reason_counter, save_path, title):
    if not reason_counter:
        return
    items = sorted(reason_counter.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in items][::-1]
    counts = [x[1] for x in items][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, counts)
    plt.xlabel("样本数")
    plt.title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_delta_bar(labels, delta_values, title, ylabel, save_path):
    plt.figure(figsize=(14, 6))
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in delta_values]
    plt.bar(labels, delta_values, color=colors)
    plt.xticks(rotation=75)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_matrix_delta(cm_exp1, cm_exp2, id2desc, save_path, title):
    delta = cm_exp2 - cm_exp1
    plt.figure(figsize=(10, 8))
    plt.imshow(delta)
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


def analyze_dataset(train_raw, test_raw, label2id, id2desc, max_len, report_dir, fig_dir):
    def build_records(raw_items, split_name):
        rows = []
        for item in raw_items:
            feat = get_sample_features(item, max_len)
            feat["数据集"] = split_name
            feat["标签编号"] = str(item["label"])
            feat["标签名称"] = id2desc[label2id[str(item["label"])]]
            rows.append(feat)
        return rows

    train_rows = build_records(train_raw, "训练集")
    test_rows = build_records(test_raw, "测试集")
    all_rows = train_rows + test_rows

    train_counts = Counter(x["标签名称"] for x in train_rows)
    test_counts = Counter(x["标签名称"] for x in test_rows)
    labels = [id2desc[i] for i in range(len(id2desc))]

    plot_bar(
        values_a=[train_counts.get(x, 0) for x in labels],
        values_b=[test_counts.get(x, 0) for x in labels],
        labels=labels,
        legend_a="训练集",
        legend_b="测试集",
        title="数据集类别分布",
        ylabel="样本数",
        save_path=fig_dir / "数据集_类别分布.png",
    )

    plot_hist(
        values_train=[x["正文分词长度"] for x in train_rows],
        values_test=[x["正文分词长度"] for x in test_rows],
        bins=30,
        title="数据集正文分词长度分布",
        xlabel="正文分词长度",
        save_path=fig_dir / "数据集_正文长度分布.png",
    )

    plot_hist(
        values_train=[x["关键词分词长度"] for x in train_rows],
        values_test=[x["关键词分词长度"] for x in test_rows],
        bins=20,
        title="数据集关键词分词长度分布",
        xlabel="关键词分词长度",
        save_path=fig_dir / "数据集_关键词长度分布.png",
    )

    plot_hist(
        values_train=[x["关键词重合率"] for x in train_rows],
        values_test=[x["关键词重合率"] for x in test_rows],
        bins=20,
        title="关键词与正文重合率分布",
        xlabel="关键词重合率",
        save_path=fig_dir / "数据集_关键词重合率分布.png",
    )

    by_class_stats = []
    for label_name in labels:
        tr = [x for x in train_rows if x["标签名称"] == label_name]
        te = [x for x in test_rows if x["标签名称"] == label_name]

        def avg(rows, key):
            if not rows:
                return 0.0
            return round(sum(r[key] for r in rows) / len(rows), 2)

        by_class_stats.append(
            {
                "类别": label_name,
                "训练集样本数": len(tr),
                "测试集样本数": len(te),
                "训练集平均正文分词长度": avg(tr, "正文分词长度"),
                "测试集平均正文分词长度": avg(te, "正文分词长度"),
                "训练集平均关键词分词长度": avg(tr, "关键词分词长度"),
                "测试集平均关键词分词长度": avg(te, "关键词分词长度"),
                "训练集平均关键词重合率": avg(tr, "关键词重合率"),
                "测试集平均关键词重合率": avg(te, "关键词重合率"),
            }
        )

    summary = {
        "训练集样本数": len(train_rows),
        "测试集样本数": len(test_rows),
        "类别数": len(labels),
        "训练集平均正文分词长度": round(np.mean([x["正文分词长度"] for x in train_rows]), 2),
        "测试集平均正文分词长度": round(np.mean([x["正文分词长度"] for x in test_rows]), 2),
        "训练集平均关键词分词长度": round(np.mean([x["关键词分词长度"] for x in train_rows]), 2),
        "测试集平均关键词分词长度": round(np.mean([x["关键词分词长度"] for x in test_rows]), 2),
        "训练集平均关键词重合率": round(np.mean([x["关键词重合率"] for x in train_rows]), 4),
        "测试集平均关键词重合率": round(np.mean([x["关键词重合率"] for x in test_rows]), 4),
        "训练集中存在截断风险的样本数": int(sum(x["存在截断风险"] for x in train_rows)),
        "测试集中存在截断风险的样本数": int(sum(x["存在截断风险"] for x in test_rows)),
        "训练集正文过短样本数": int(sum(x["正文过短"] for x in train_rows)),
        "测试集正文过短样本数": int(sum(x["正文过短"] for x in test_rows)),
    }

    save_json(summary, report_dir / "数据集概览.json")
    save_json(by_class_stats, report_dir / "数据集_按类别统计.json")
    save_json(all_rows[:300], report_dir / "数据集_样本特征预览.json")

    return {
        "摘要": summary,
        "按类别统计": by_class_stats,
    }


def build_test_loader(config, processed_dir, raw_dir):
    vocab_obj = load_json(processed_dir / "vocab.json")
    token2id = vocab_obj["token2id"]
    pad_idx = token2id[vocab_obj.get("pad_token", "[PAD]")]
    vocab_size = len(token2id)

    test_seq = torch.load(processed_dir / "test_seq.pt")
    test_kw_seq = torch.load(processed_dir / "test_kw_seq.pt")

    label2id, id2desc = load_label_mapping(raw_dir / "labels.json")
    test_raw = read_jsonl(raw_dir / "test.json")
    test_labels = torch.tensor([label2id[str(x["label"])] for x in test_raw], dtype=torch.long)

    return {
        "词表对象": vocab_obj,
        "词表大小": vocab_size,
        "填充编号": pad_idx,
        "测试正文序列": test_seq,
        "测试关键词序列": test_kw_seq,
        "测试原始样本": test_raw,
        "测试标签": test_labels,
        "标签到编号": label2id,
        "编号到类别": id2desc,
    }


def save_prediction_details(raw_items, true_ids, pred_ids, confs, id2desc, max_len, save_path):
    rows = []
    for item, y_true, y_pred, conf in zip(raw_items, true_ids, pred_ids, confs):
        feat = get_sample_features(item, max_len)
        rows.append(
            {
                **feat,
                "真实类别": id2desc[y_true],
                "预测类别": id2desc[y_pred],
                "是否预测正确": bool(y_true == y_pred),
                "模型置信度": round(float(conf), 4),
            }
        )
    save_json(rows, save_path)


def analyze_error_reasons(raw_items, true_ids, pred_ids, confs, id2desc, max_len, use_keywords, top_confusion_pairs):
    top_pair_set = set()
    for _, pair_text in top_confusion_pairs:
        left, right = pair_text.split(" -> ")
        top_pair_set.add((left, right))

    reason_counter = Counter()
    reason_cases = defaultdict(list)

    for item, y_true, y_pred, conf in zip(raw_items, true_ids, pred_ids, confs):
        if y_true == y_pred:
            continue

        feat = get_sample_features(item, max_len)
        true_name = id2desc[y_true]
        pred_name = id2desc[y_pred]
        reasons = []

        if feat["存在截断风险"]:
            reasons.append("正文过长可能被截断")
        if feat["正文过短"]:
            reasons.append("正文过短信息不足")
        if (true_name, pred_name) in top_pair_set:
            reasons.append("类别语义接近容易混淆")
        if float(conf) < 0.45:
            reasons.append("模型置信度较低")
        if use_keywords and feat["关键词缺失或极少"]:
            reasons.append("关键词缺失或过少")
        if use_keywords and feat["关键词重合度低"]:
            reasons.append("关键词与正文重合度低")

        if not reasons:
            reasons.append("其他复杂错误")

        main_reason = reasons[0]
        reason_counter[main_reason] += 1

        if len(reason_cases[main_reason]) < 20:
            reason_cases[main_reason].append(
                {
                    "正文": item.get("sentence", ""),
                    "关键词": item.get("keywords", ""),
                    "真实类别": true_name,
                    "预测类别": pred_name,
                    "模型置信度": round(float(conf), 4),
                    "正文分词长度": feat["正文分词长度"],
                    "关键词分词长度": feat["关键词分词长度"],
                    "关键词重合率": feat["关键词重合率"],
                    "原因列表": reasons,
                }
            )

    return dict(reason_counter), dict(reason_cases)


def analyze_single_experiment(exp_name, config, raw_dir, processed_dir, report_dir, fig_dir):
    built = build_test_loader(config, processed_dir, raw_dir)
    vocab_size = built["词表大小"]
    pad_idx = built["填充编号"]
    test_seq = built["测试正文序列"]
    test_kw_seq = built["测试关键词序列"]
    test_raw = built["测试原始样本"]
    test_labels = built["测试标签"]
    id2desc = built["编号到类别"]

    use_keywords = exp_name == "exp2"
    dataset = NewsDataset(
        seqs=test_seq,
        labels=test_labels,
        keyword_seqs=test_kw_seq if use_keywords else None,
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = build_model(exp_name, config, vocab_size, pad_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_path = Path(config["paths"]["output_dir"]) / "models" / f"best_model_{exp_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    true_ids, pred_ids, confs = predict(model, loader, device, use_keywords)

    names = [id2desc[i] for i in range(len(id2desc))]
    acc = accuracy_score(true_ids, pred_ids)
    macro_f1 = f1_score(true_ids, pred_ids, average="macro", zero_division=0)
    report_text = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(names))),
        target_names=names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(true_ids, pred_ids, labels=list(range(len(names))))

    top_confusion_pairs = plot_top_confusions(
        cm=cm,
        id2desc=id2desc,
        save_path=fig_dir / f"{exp_name}_高频错分对.png",
        topn=10,
    )

    reason_counter, reason_cases = analyze_error_reasons(
        raw_items=test_raw,
        true_ids=true_ids,
        pred_ids=pred_ids,
        confs=confs,
        id2desc=id2desc,
        max_len=int(config["max_len"]),
        use_keywords=use_keywords,
        top_confusion_pairs=top_confusion_pairs,
    )

    metrics_obj = {
        "实验名称": exp_name,
        "是否使用关键词": bool(use_keywords),
        "测试集样本数": int(len(true_ids)),
        "准确率": round(float(acc), 6),
        "宏平均F1": round(float(macro_f1), 6),
    }

    save_json(metrics_obj, report_dir / f"{exp_name}_指标.json")
    save_text(report_text, report_dir / f"{exp_name}_分类报告.txt")
    save_prediction_details(
        raw_items=test_raw,
        true_ids=true_ids,
        pred_ids=pred_ids,
        confs=confs,
        id2desc=id2desc,
        max_len=int(config["max_len"]),
        save_path=report_dir / f"{exp_name}_预测明细.json",
    )

    badcases = []
    for item, y_true, y_pred, conf in zip(test_raw, true_ids, pred_ids, confs):
        if y_true != y_pred:
            feat = get_sample_features(item, int(config["max_len"]))
            badcases.append(
                {
                    **feat,
                    "真实类别": id2desc[y_true],
                    "预测类别": id2desc[y_pred],
                    "模型置信度": round(float(conf), 4),
                }
            )
    save_json(badcases, report_dir / f"{exp_name}_错分样本.json")
    save_json(reason_counter, report_dir / f"{exp_name}_错分原因统计.json")
    save_json(reason_cases, report_dir / f"{exp_name}_错分原因样例.json")

    plot_confusion(
        cm=cm,
        id2desc=id2desc,
        save_path=fig_dir / f"{exp_name}_混淆矩阵.png",
        title=f"{exp_name} 测试集混淆矩阵",
    )
    plot_per_class_metrics(
        report_dict=report_dict,
        id2desc=id2desc,
        save_path=fig_dir / f"{exp_name}_各类别指标.png",
        title=f"{exp_name} 各类别 Precision / Recall / F1",
    )
    plot_length_bucket_accuracy(
        raw_items=test_raw,
        true_ids=true_ids,
        pred_ids=pred_ids,
        field_name="sentence",
        save_path=fig_dir / f"{exp_name}_正文长度分桶准确率.png",
        title=f"{exp_name} 按正文长度分桶的准确率",
    )
    if use_keywords:
        plot_length_bucket_accuracy(
            raw_items=test_raw,
            true_ids=true_ids,
            pred_ids=pred_ids,
            field_name="keywords",
            save_path=fig_dir / f"{exp_name}_关键词长度分桶准确率.png",
            title=f"{exp_name} 按关键词长度分桶的准确率",
        )
    plot_reason_counts(
        reason_counter=reason_counter,
        save_path=fig_dir / f"{exp_name}_错分原因统计.png",
        title=f"{exp_name} 错分原因统计",
    )

    return {
        "实验名称": exp_name,
        "原始样本": test_raw,
        "真实标签": true_ids,
        "预测标签": pred_ids,
        "置信度": confs,
        "分类报告字典": report_dict,
        "混淆矩阵": cm,
        "编号到类别": id2desc,
        "总体指标": metrics_obj,
    }


def compare_experiments(result_exp1, result_exp2, report_dir, fig_dir):
    id2desc = result_exp1["编号到类别"]
    names = [id2desc[i] for i in range(len(id2desc))]

    metrics_compare = {
        "实验一_准确率": result_exp1["总体指标"]["准确率"],
        "实验二_准确率": result_exp2["总体指标"]["准确率"],
        "实验一_宏平均F1": result_exp1["总体指标"]["宏平均F1"],
        "实验二_宏平均F1": result_exp2["总体指标"]["宏平均F1"],
        "准确率提升": round(
            result_exp2["总体指标"]["准确率"] - result_exp1["总体指标"]["准确率"], 6
        ),
        "宏平均F1提升": round(
            result_exp2["总体指标"]["宏平均F1"] - result_exp1["总体指标"]["宏平均F1"], 6
        ),
    }
    save_json(metrics_compare, report_dir / "实验对比_总体指标.json")

    plot_bar(
        values_a=[
            result_exp1["总体指标"]["准确率"],
            result_exp1["总体指标"]["宏平均F1"],
        ],
        values_b=[
            result_exp2["总体指标"]["准确率"],
            result_exp2["总体指标"]["宏平均F1"],
        ],
        labels=["准确率", "宏平均F1"],
        legend_a="实验一：只用正文",
        legend_b="实验二：正文+关键词",
        title="实验一与实验二总体指标对比",
        ylabel="分数",
        save_path=fig_dir / "实验对比_总体指标.png",
        rotation=0,
    )

    per_class_rows = []
    delta_f1 = []
    delta_recall = []

    for name in names:
        p1 = result_exp1["分类报告字典"].get(name, {})
        p2 = result_exp2["分类报告字典"].get(name, {})
        row = {
            "类别": name,
            "实验一_Precision": round(float(p1.get("precision", 0.0)), 6),
            "实验一_Recall": round(float(p1.get("recall", 0.0)), 6),
            "实验一_F1": round(float(p1.get("f1-score", 0.0)), 6),
            "实验二_Precision": round(float(p2.get("precision", 0.0)), 6),
            "实验二_Recall": round(float(p2.get("recall", 0.0)), 6),
            "实验二_F1": round(float(p2.get("f1-score", 0.0)), 6),
            "Recall提升": round(float(p2.get("recall", 0.0) - p1.get("recall", 0.0)), 6),
            "F1提升": round(float(p2.get("f1-score", 0.0) - p1.get("f1-score", 0.0)), 6),
        }
        per_class_rows.append(row)
        delta_f1.append(row["F1提升"])
        delta_recall.append(row["Recall提升"])

    per_class_rows = sorted(per_class_rows, key=lambda x: x["F1提升"], reverse=True)
    save_json(per_class_rows, report_dir / "实验对比_各类别差异.json")

    plot_delta_bar(
        labels=names,
        delta_values=[
            result_exp2["分类报告字典"].get(name, {}).get("f1-score", 0.0)
            - result_exp1["分类报告字典"].get(name, {}).get("f1-score", 0.0)
            for name in names
        ],
        title="实验二相对实验一的各类别 F1 提升",
        ylabel="F1 提升值",
        save_path=fig_dir / "实验对比_各类别F1提升.png",
    )

    plot_delta_bar(
        labels=names,
        delta_values=[
            result_exp2["分类报告字典"].get(name, {}).get("recall", 0.0)
            - result_exp1["分类报告字典"].get(name, {}).get("recall", 0.0)
            for name in names
        ],
        title="实验二相对实验一的各类别 Recall 提升",
        ylabel="Recall 提升值",
        save_path=fig_dir / "实验对比_各类别Recall提升.png",
    )

    plot_matrix_delta(
        cm_exp1=result_exp1["混淆矩阵"],
        cm_exp2=result_exp2["混淆矩阵"],
        id2desc=id2desc,
        save_path=fig_dir / "实验对比_混淆矩阵差值.png",
        title="实验二减去实验一的混淆矩阵差值",
    )

    improved_cases = []
    degraded_cases = []
    same_raw = result_exp1["原始样本"]

    for item, y_true, p1, p2, c1, c2 in zip(
        same_raw,
        result_exp1["真实标签"],
        result_exp1["预测标签"],
        result_exp2["预测标签"],
        result_exp1["置信度"],
        result_exp2["置信度"],
    ):
        feat = get_sample_features(item, 999999)
        base = {
            "正文": item.get("sentence", ""),
            "关键词": item.get("keywords", ""),
            "真实类别": id2desc[y_true],
            "实验一预测": id2desc[p1],
            "实验二预测": id2desc[p2],
            "实验一置信度": round(float(c1), 4),
            "实验二置信度": round(float(c2), 4),
            "正文分词长度": feat["正文分词长度"],
            "关键词分词长度": feat["关键词分词长度"],
            "关键词重合率": feat["关键词重合率"],
        }

        if p1 != y_true and p2 == y_true:
            improved_cases.append(base)
        elif p1 == y_true and p2 != y_true:
            degraded_cases.append(base)

    save_json(improved_cases, report_dir / "实验对比_exp2改进样本.json")
    save_json(degraded_cases, report_dir / "实验对比_exp2退化样本.json")

    summary_text = "\n".join(
        [
            "实验一与实验二对比总结",
            "=" * 40,
            f"实验一 准确率: {result_exp1['总体指标']['准确率']:.4f}",
            f"实验二 准确率: {result_exp2['总体指标']['准确率']:.4f}",
            f"准确率提升: {metrics_compare['准确率提升']:.4f}",
            "",
            f"实验一 宏平均F1: {result_exp1['总体指标']['宏平均F1']:.4f}",
            f"实验二 宏平均F1: {result_exp2['总体指标']['宏平均F1']:.4f}",
            f"宏平均F1提升: {metrics_compare['宏平均F1提升']:.4f}",
            "",
            f"实验二修正成功的样本数: {len(improved_cases)}",
            f"实验二反而退化的样本数: {len(degraded_cases)}",
            "",
            "各类别差异明细已保存到：实验对比_各类别差异.json",
        ]
    )
    save_text(summary_text, report_dir / "实验对比_总结.txt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["all", "exp1", "exp2"],
        help="all 表示同时做数据集分析、实验一分析、实验二分析和实验对比",
    )
    return parser.parse_args()


def main():
    setup_chinese_font()
    args = parse_args()

    config = load_yaml(Path("config.yaml"))
    set_seed(int(config["seed"]))

    raw_dir = resolve_raw_dir(config)
    processed_dir = Path(config["paths"]["processed_dir"])
    output_dir = Path(config["paths"]["output_dir"])

    report_dir = output_dir / "reports"
    fig_dir = output_dir / "figs"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_raw = read_jsonl(raw_dir / "train.json")
    test_raw = read_jsonl(raw_dir / "test.json")
    label2id, id2desc = load_label_mapping(raw_dir / "labels.json")

    analyze_dataset(
        train_raw=train_raw,
        test_raw=test_raw,
        label2id=label2id,
        id2desc=id2desc,
        max_len=int(config["max_len"]),
        report_dir=report_dir,
        fig_dir=fig_dir,
    )

    result_exp1 = None
    result_exp2 = None

    if args.exp in ["all", "exp1"]:
        result_exp1 = analyze_single_experiment(
            exp_name="exp1",
            config=config,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            report_dir=report_dir,
            fig_dir=fig_dir,
        )

    if args.exp in ["all", "exp2"]:
        result_exp2 = analyze_single_experiment(
            exp_name="exp2",
            config=config,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            report_dir=report_dir,
            fig_dir=fig_dir,
        )

    if args.exp == "all":
        if result_exp1 is None or result_exp2 is None:
            raise RuntimeError("实验对比需要同时分析 exp1 和 exp2")
        compare_experiments(
            result_exp1=result_exp1,
            result_exp2=result_exp2,
            report_dir=report_dir,
            fig_dir=fig_dir,
        )

    print("=" * 60)
    print("分析完成")
    print(f"图表目录：{fig_dir}")
    print(f"报告目录：{report_dir}")


if __name__ == "__main__":
    main()