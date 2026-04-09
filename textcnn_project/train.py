import torch
from torch.utils.data import Dataset

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
            "label": self.labels[idx]
        }
        if self.key_masks is not None:
            item["key_mask"] = self.key_masks[idx]
        return item