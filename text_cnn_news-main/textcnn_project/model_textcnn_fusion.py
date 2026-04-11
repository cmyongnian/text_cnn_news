from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeywordFusionTextCNN(nn.Module):
    """
    实验二模型：
    仍然以 TextCNN 为主体，但在 embedding 层对关键词位置做增强。
    key_mask 来自 preprocess.py 生成的关键词位置掩码。
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 200,
        num_filters: int = 100,
        kernel_sizes=(3, 4, 5),
        dropout: float = 0.5,
        pad_idx: int = 0,
        keyword_scale: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 学习一个关键词偏置向量；关键词位置会额外叠加这个偏置
        self.keyword_bias = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.keyword_scale = keyword_scale

        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, key_mask: torch.Tensor):
        # input_ids: [B, L]
        # key_mask:  [B, L]，0/1
        x = self.embedding(input_ids)  # [B, L, E]

        key_mask = key_mask.unsqueeze(-1).float()  # [B, L, 1]
        x = x + self.keyword_scale * key_mask * self.keyword_bias

        x = x.transpose(1, 2)  # [B, E, L]

        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]

        feat = torch.cat(pooled, dim=1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits