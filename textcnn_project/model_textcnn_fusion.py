import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 200,
        num_filters: int = 128,
        kernel_sizes=(2, 3, 4, 5),
        dropout: float = 0.5,
        emb_dropout: float = 0.1,
        fusion_hidden_dim: int = 128,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.sent_convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.kw_convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )

        feat_dim = num_filters * len(kernel_sizes)

        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def _conv_pool(self, emb: torch.Tensor, convs: nn.ModuleList) -> torch.Tensor:
        x = emb.transpose(1, 2)
        features = []
        for conv in convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            features.append(h)
        return torch.cat(features, dim=1)

    def forward(self, input_ids: torch.Tensor, keyword_ids: torch.Tensor) -> torch.Tensor:
        sent_emb = self.emb_dropout(self.embedding(input_ids))
        kw_emb = self.emb_dropout(self.embedding(keyword_ids))

        sent_feat = self._conv_pool(sent_emb, self.sent_convs)
        kw_feat = self._conv_pool(kw_emb, self.kw_convs)

        gate = self.gate(torch.cat([sent_feat, kw_feat], dim=1))
        kw_feat = gate * kw_feat
        fused = torch.cat([sent_feat, kw_feat], dim=1)
        logits = self.classifier(fused)
        return logits