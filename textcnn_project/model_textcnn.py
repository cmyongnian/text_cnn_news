import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=200,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        dropout=0.5,
        pad_idx=0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids):
        # input_ids: [B, L]
        x = self.embedding(input_ids)      # [B, L, E]
        x = x.transpose(1, 2)              # [B, E, L]

        conv_outs = [F.relu(conv(x)) for conv in self.convs]   # list of [B, C, L-k+1]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]  # list of [B, C]

        feat = torch.cat(pooled, dim=1)    # [B, 3*C]
        feat = self.dropout(feat)
        logits = self.fc(feat)             # [B, num_classes]
        return logits