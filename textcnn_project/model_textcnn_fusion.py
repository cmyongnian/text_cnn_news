import torch
import torch.nn as nn
import torch.nn.functional as F

class KeywordFusionTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=200,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        dropout=0.5,
        pad_idx=0,
        keyword_scale=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.keyword_bias = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.keyword_scale = keyword_scale

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids, key_mask):
        # input_ids: [B, L]
        # key_mask:  [B, L], 值是0/1
        x = self.embedding(input_ids)  # [B, L, E]

        # 在关键词位置加偏置
        key_mask = key_mask.unsqueeze(-1).float()   # [B, L, 1]
        x = x + self.keyword_scale * key_mask * self.keyword_bias

        x = x.transpose(1, 2)  # [B, E, L]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        feat = torch.cat(pooled, dim=1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits