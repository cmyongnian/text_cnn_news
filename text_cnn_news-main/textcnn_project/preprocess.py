from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import jieba
import torch

from utils import PAD_TOKEN, UNK_TOKEN, ensure_dir, load_yaml, read_jsonl, save_json, save_text


def parse_args():
    parser = argparse.ArgumentParser(description="文本预处理：分词、建词表、序列化、关键词掩码")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def normalize_text(text) -> str:
    if text is None:
        return ""
    return str(text).strip()


def clean_tokens(tokens):
    return [tok.strip() for tok in tokens if tok and tok.strip()]


def tokenize_sentence(text: str):
    return clean_tokens(jieba.lcut(normalize_text(text), cut_all=False))


def tokenize_keywords(keywords: str):
    keywords = normalize_text(keywords)
    if not keywords:
        return []

    parts = [p for p in re.split(r"[，,；;、\s]+", keywords) if p.strip()]
    tokens = []
    for part in parts:
        tokens.extend(clean_tokens(jieba.lcut(part, cut_all=False)))
    return tokens


def prepare_samples(data):
    samples = []
    for item in data:
        sentence_tokens = tokenize_sentence(item.get("sentence", ""))
        keyword_tokens = tokenize_keywords(item.get("keywords", ""))

        samples.append(
            {
                "label": str(item["label"]),
                "label_desc": item.get("label_desc", ""),
                "sentence": item.get("sentence", ""),
                "keywords": item.get("keywords", ""),
                "sentence_tokens": sentence_tokens,
                "keyword_tokens": keyword_tokens,
            }
        )
    return samples


def build_vocab(train_samples, max_vocab_size: int):
    """
    词表只根据训练集 sentence 构建，避免信息泄漏。
    """
    counter = Counter()
    for item in train_samples:
        counter.update(item["sentence_tokens"])

    token2id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, _ in counter.most_common(max(0, max_vocab_size - 2)):
        if token not in token2id:
            token2id[token] = len(token2id)

    return token2id


def encode_tokens(tokens, token2id, max_len: int):
    unk_id = token2id[UNK_TOKEN]
    pad_id = token2id[PAD_TOKEN]

    ids = [token2id.get(tok, unk_id) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([pad_id] * (max_len - len(ids)))
    return ids


def build_key_mask(sentence_tokens, keyword_tokens, max_len: int):
    """
    对 sentence 中命中的关键词 token 标 1，否则标 0。
    """
    keyword_set = set(keyword_tokens)
    mask = [1 if tok in keyword_set else 0 for tok in sentence_tokens[:max_len]]
    if len(mask) < max_len:
        mask.extend([0] * (max_len - len(mask)))
    return mask


def convert_samples_to_features(samples, token2id, max_len: int):
    seqs = []
    key_masks = []
    export_tokens = []

    for item in samples:
        sentence_tokens = item["sentence_tokens"]
        keyword_tokens = item["keyword_tokens"]

        seq = encode_tokens(sentence_tokens, token2id, max_len)
        key_mask = build_key_mask(sentence_tokens, keyword_tokens, max_len)

        seqs.append(seq)
        key_masks.append(key_mask)

        export_tokens.append(
            {
                "label": item["label"],
                "label_desc": item["label_desc"],
                "sentence_tokens": sentence_tokens,
                "keyword_tokens": keyword_tokens,
            }
        )

    seq_tensor = torch.tensor(seqs, dtype=torch.long)
    key_mask_tensor = torch.tensor(key_masks, dtype=torch.long)
    return seq_tensor, key_mask_tensor, export_tokens


def build_statistics_text(train_samples, test_samples, token2id, max_len: int):
    train_lengths = [len(x["sentence_tokens"]) for x in train_samples]
    test_lengths = [len(x["sentence_tokens"]) for x in test_samples]

    train_keyword_hits = [
        sum(build_key_mask(x["sentence_tokens"], x["keyword_tokens"], max_len))
        for x in train_samples
    ]
    test_keyword_hits = [
        sum(build_key_mask(x["sentence_tokens"], x["keyword_tokens"], max_len))
        for x in test_samples
    ]

    train_empty_keywords = sum(1 for x in train_samples if len(x["keyword_tokens"]) == 0)
    test_empty_keywords = sum(1 for x in test_samples if len(x["keyword_tokens"]) == 0)

    lines = [
        "数据统计",
        "=" * 40,
        f"train_samples: {len(train_samples)}",
        f"test_samples: {len(test_samples)}",
        f"vocab_size: {len(token2id)}",
        f"max_len: {max_len}",
        "",
        "句长统计",
        "-" * 40,
        f"train_avg_len: {mean(train_lengths):.2f}" if train_lengths else "train_avg_len: 0",
        f"train_max_len: {max(train_lengths) if train_lengths else 0}",
        f"test_avg_len: {mean(test_lengths):.2f}" if test_lengths else "test_avg_len: 0",
        f"test_max_len: {max(test_lengths) if test_lengths else 0}",
        "",
        "关键词统计",
        "-" * 40,
        f"train_empty_keywords: {train_empty_keywords}",
        f"test_empty_keywords: {test_empty_keywords}",
        f"train_avg_keyword_hits: {mean(train_keyword_hits):.2f}" if train_keyword_hits else "train_avg_keyword_hits: 0",
        f"test_avg_keyword_hits: {mean(test_keyword_hits):.2f}" if test_keyword_hits else "test_avg_keyword_hits: 0",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_yaml(project_root / args.config)

    raw_dir = project_root / config["paths"]["raw_dir"]
    processed_dir = project_root / config["paths"]["processed_dir"]

    ensure_dir(processed_dir)

    train_path = raw_dir / "train.json"
    test_path = raw_dir / "test.json"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"未找到原始数据文件。\n"
            f"当前读取目录: {raw_dir}\n"
            f"请确认 config.yaml 中 paths.raw_dir 指向包含 train.json / test.json / labels.json 的目录。"
        )

    train_raw = read_jsonl(train_path)
    test_raw = read_jsonl(test_path)

    train_samples = prepare_samples(train_raw)
    test_samples = prepare_samples(test_raw)

    token2id = build_vocab(train_samples, config["vocab_size"])

    train_seq, train_key_mask, train_tokens_export = convert_samples_to_features(
        train_samples, token2id, config["max_len"]
    )
    test_seq, test_key_mask, test_tokens_export = convert_samples_to_features(
        test_samples, token2id, config["max_len"]
    )

    vocab_obj = {
        "token2id": token2id,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "max_len": config["max_len"],
    }

    save_json(processed_dir / "vocab.json", vocab_obj)
    save_json(processed_dir / "train_tokens.json", train_tokens_export)
    save_json(processed_dir / "test_tokens.json", test_tokens_export)

    torch.save(train_seq, processed_dir / "train_seq.pt")
    torch.save(test_seq, processed_dir / "test_seq.pt")
    torch.save(train_key_mask, processed_dir / "train_key_mask.pt")
    torch.save(test_key_mask, processed_dir / "test_key_mask.pt")

    statistics_text = build_statistics_text(
        train_samples=train_samples,
        test_samples=test_samples,
        token2id=token2id,
        max_len=config["max_len"],
    )
    save_text(processed_dir / "data_statistics.txt", statistics_text)

    print("预处理完成，已生成：")
    print(processed_dir / "vocab.json")
    print(processed_dir / "train_tokens.json")
    print(processed_dir / "test_tokens.json")
    print(processed_dir / "train_seq.pt")
    print(processed_dir / "test_seq.pt")
    print(processed_dir / "train_key_mask.pt")
    print(processed_dir / "test_key_mask.pt")
    print(processed_dir / "data_statistics.txt")


if __name__ == "__main__":
    main()