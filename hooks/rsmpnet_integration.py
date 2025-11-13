import os
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.pose_fusion_module import PoseFiLMBlock, PoseGate
from modules.consistency_teacher import consistency_loss
from .common_metrics import MapMetricsAggregator


class RSMPDecoderWithPose(nn.Module):
    def __init__(self, chs: Tuple[int, int, int, int] = (256, 128, 64, 32),
                 num_classes: int = 21, pose_dim: int = 3, mode: str = "film"):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(chs[0], chs[1], 2, 2)
        self.g1 = PoseGate(chs[1], pose_dim)
        self.b1 = PoseFiLMBlock(chs[1], chs[1], pose_dim=pose_dim, mode=mode)
        self.up2 = nn.ConvTranspose2d(chs[1], chs[2], 2, 2)
        self.g2 = PoseGate(chs[2], pose_dim)
        self.b2 = PoseFiLMBlock(chs[2], chs[2], pose_dim=pose_dim, mode=mode)
        self.up3 = nn.ConvTranspose2d(chs[2], chs[3], 2, 2)
        self.g3 = PoseGate(chs[3], pose_dim)
        self.b3 = PoseFiLMBlock(chs[3], chs[3], pose_dim=pose_dim, mode=mode)
        self.head = nn.Conv2d(chs[3], num_classes, 1)

    def forward(self, feats: torch.Tensor, skips: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], pose: torch.Tensor) -> torch.Tensor:
        x = self.up1(feats)
        x = self.g1(skips[0], x, pose)
        x = self.b1(x, pose)
        x = self.up2(x)
        x = self.g2(skips[1], x, pose)
        x = self.b2(x, pose)
        x = self.up3(x)
        x = self.g3(skips[2], x, pose)
        x = self.b3(x, pose)
        return self.head(x)


def rsmp_kd_loss(student_logits: torch.Tensor, teacher_prob: Optional[torch.Tensor],
                 valid_mask: Optional[torch.Tensor], kd_w: float = 0.25, mode: str = "l1") -> torch.Tensor:
    if teacher_prob is None:
        return torch.tensor(0.0, dtype=student_logits.dtype, device=student_logits.device)
    return consistency_loss(student_logits, teacher_prob, valid_mask, w=kd_w, mode=mode,
                            student_is_logit=True, teacher_is_logit=False)


class RSMPIntegrationHelper:
    def __init__(self, num_classes: int = 21, pose_dim: int = 3, kd_w: float = 0.25, kd_mode: str = "l1"):
        self.num_classes = num_classes
        self.pose_dim = pose_dim
        self.kd_w = kd_w
        self.kd_mode = kd_mode
        self.decoder: Optional[RSMPDecoderWithPose] = None
        self.agg = MapMetricsAggregator(num_classes)

    def build(self, chs: Tuple[int, int, int, int] = (256, 128, 64, 32), device: Optional[torch.device] = None) -> RSMPDecoderWithPose:
        self.decoder = RSMPDecoderWithPose(chs=chs, num_classes=self.num_classes, pose_dim=self.pose_dim, mode="film")
        if device is not None:
            self.decoder.to(device)
        return self.decoder

    def loss(self, logits: torch.Tensor, target: torch.Tensor,
             teacher_prob: Optional[torch.Tensor] = None, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, ignore_index=-1)
        kd = rsmp_kd_loss(logits, teacher_prob, valid_mask, kd_w=self.kd_w, mode=self.kd_mode)
        return ce + kd

    def update_metrics(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        self.agg.update(pred, gt)

    def compute_metrics(self) -> Dict[str, float]:
        return self.agg.compute()
