import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 200,
        num_filters: int = 128,
        kernel_sizes=(2, 3, 4, 5),
        dropout: float = 0.5,
        emb_dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def _conv_pool(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            features.append(h)
        return torch.cat(features, dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.emb_dropout(x)
        feat = self._conv_pool(x)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits