"""Small learned BEV model used by Course 05 and the capstone."""

from __future__ import annotations

import torch
from torch import nn


class LearnableBEVModel(nn.Module):
    """Sensor tokens + spatial BEV queries -> occupancy/risk/velocity heads."""

    def __init__(self, height: int, width: int, in_channels: int = 3,
                 d_model: int = 48, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.height = height
        self.width = width
        self.num_cells = height * width
        self.input_proj = nn.Linear(in_channels, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.num_cells, d_model))
        self.bev_queries = nn.Parameter(torch.zeros(1, self.num_cells, d_model))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.bev_queries, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.sensor_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        if (height, width) != (self.height, self.width):
            raise ValueError(f"expected BEV {(self.height, self.width)}, got {(height, width)}")
        tokens = features.permute(0, 2, 3, 1).reshape(batch, self.num_cells, channels)
        tokens = self.input_proj(tokens) + self.position
        memory = self.sensor_encoder(tokens)
        queries = self.bev_queries.expand(batch, -1, -1) + self.position
        fused, _ = self.cross_attention(queries, memory, memory, need_weights=False)
        return self.head(fused).reshape(batch, self.height, self.width, 3).permute(0, 3, 1, 2)


def occupancy_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    prediction = (torch.sigmoid(logits) >= threshold)
    truth = target.bool()
    intersection = (prediction & truth).sum().item()
    union = (prediction | truth).sum().item()
    return float(intersection / max(union, 1))


def risk_f1(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    prediction = (torch.sigmoid(logits) >= threshold)
    truth = target.bool()
    tp = (prediction & truth).sum().item()
    fp = (prediction & ~truth).sum().item()
    fn = (~prediction & truth).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))
