import json

def load_label_mapping(label_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            labels.append(obj["label"])
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data