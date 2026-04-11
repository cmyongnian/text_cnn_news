import json
import re
from collections import Counter
from pathlib import Path

import jieba
import torch
import yaml

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def prepare_samples(raw_items):
    samples = []
    for item in raw_items:
        sentence = normalize_text(item.get("sentence", ""))
        keywords = normalize_text(item.get("keywords", ""))
        samples.append(
            {
                "label": str(item["label"]),
                "sentence": sentence,
                "keywords": keywords,
                "sentence_tokens": tokenize_sentence(sentence),
                "keyword_tokens": tokenize_keywords(keywords),
            }
        )
    return samples


def build_vocab(train_samples, max_vocab_size: int, min_freq: int):
    counter = Counter()
    for item in train_samples:
        counter.update(item["sentence_tokens"])

    token2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token not in token2id:
            token2id[token] = len(token2id)
        if len(token2id) >= max_vocab_size:
            break
    return token2id


def encode_tokens(tokens, token2id, max_len: int):
    unk_id = token2id[UNK_TOKEN]
    pad_id = token2id[PAD_TOKEN]
    ids = [token2id.get(tok, unk_id) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([pad_id] * (max_len - len(ids)))
    return ids


def convert_samples_to_tensors(samples, token2id, max_len: int, max_keyword_len: int):
    seqs = []
    kw_seqs = []
    preview = []

    for item in samples:
        sent_ids = encode_tokens(item["sentence_tokens"], token2id, max_len)
        kw_ids = encode_tokens(item["keyword_tokens"], token2id, max_keyword_len)
        seqs.append(sent_ids)
        kw_seqs.append(kw_ids)

        preview.append(
            {
                "label": item["label"],
                "sentence": item["sentence"],
                "keywords": item["keywords"],
                "sentence_tokens": item["sentence_tokens"][:30],
                "keyword_tokens": item["keyword_tokens"][:20],
                "sentence_len": len(item["sentence_tokens"]),
                "keyword_len": len(item["keyword_tokens"]),
            }
        )

    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(kw_seqs, dtype=torch.long),
        preview,
    )


def main():
    config = load_yaml(Path("config.yaml"))
    raw_dir = resolve_raw_dir(config)
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_raw = read_jsonl(raw_dir / "train.json")
    test_raw = read_jsonl(raw_dir / "test.json")

    train_samples = prepare_samples(train_raw)
    test_samples = prepare_samples(test_raw)

    token2id = build_vocab(
        train_samples=train_samples,
        max_vocab_size=int(config["vocab_size"]),
        min_freq=int(config.get("min_freq", 1)),
    )

    max_len = int(config["max_len"])
    max_keyword_len = int(config.get("max_keyword_len", 16))

    train_seq, train_kw_seq, train_preview = convert_samples_to_tensors(
        train_samples, token2id, max_len, max_keyword_len
    )
    test_seq, test_kw_seq, test_preview = convert_samples_to_tensors(
        test_samples, token2id, max_len, max_keyword_len
    )

    torch.save(train_seq, processed_dir / "train_seq.pt")
    torch.save(test_seq, processed_dir / "test_seq.pt")
    torch.save(train_kw_seq, processed_dir / "train_kw_seq.pt")
    torch.save(test_kw_seq, processed_dir / "test_kw_seq.pt")

    vocab_obj = {
        "token2id": token2id,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "max_len": max_len,
        "max_keyword_len": max_keyword_len,
        "vocab_size": len(token2id),
        "min_freq": int(config.get("min_freq", 1)),
    }
    save_json(vocab_obj, processed_dir / "vocab.json")
    save_json(train_preview[:200], processed_dir / "train_preview.json")
    save_json(test_preview[:200], processed_dir / "test_preview.json")

    summary = {
        "raw_dir": str(raw_dir),
        "num_train": len(train_samples),
        "num_test": len(test_samples),
        "vocab_size": len(token2id),
        "max_len": max_len,
        "max_keyword_len": max_keyword_len,
        "avg_train_sentence_len": round(
            sum(len(x["sentence_tokens"]) for x in train_samples) / max(1, len(train_samples)), 2
        ),
        "avg_test_sentence_len": round(
            sum(len(x["sentence_tokens"]) for x in test_samples) / max(1, len(test_samples)), 2
        ),
        "avg_train_keyword_len": round(
            sum(len(x["keyword_tokens"]) for x in train_samples) / max(1, len(train_samples)), 2
        ),
        "avg_test_keyword_len": round(
            sum(len(x["keyword_tokens"]) for x in test_samples) / max(1, len(test_samples)), 2
        ),
    }
    save_json(summary, processed_dir / "data_summary.json")
    print("预处理完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()