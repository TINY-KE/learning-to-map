import os
import csv
import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.pose_fusion_module import PoseFiLMBlock, PoseGate, PoseConditionLayer
from modules.active_training_reward import compute_active_reward, batch_active_reward
from modules.consistency_teacher import consistency_loss
from modules.geometry_utils import (
    CameraIntrinsics,
    letterbox_pad_torch,
    resize_image_with_intrinsics,
    pack_intrinsics,
)

from .common_metrics import (
    success_rate,
    spl_score,
    soft_spl_score,
    dts_score,
    MapMetricsAggregator,
    MetricsLogger,
)


class L2MDecoderWithPose(nn.Module):
    def __init__(self, in_ch: int, skip_chs: Tuple[int, int, int], out_chs: Tuple[int, int, int],
                 num_classes: int = 21, pose_dim: int = 3, mode: str = "film"):
        super().__init__()
        assert len(skip_chs) == 3 and len(out_chs) == 3
        self.up1 = nn.ConvTranspose2d(in_ch, out_chs[0], 2, stride=2)
        self.g1 = PoseGate(skip_chs[0], pose_dim)
        self.b1 = PoseFiLMBlock(out_chs[0], out_chs[0], pose_dim=pose_dim, mode=mode)
        self.up2 = nn.ConvTranspose2d(out_chs[0], out_chs[1], 2, stride=2)
        self.g2 = PoseGate(skip_chs[1], pose_dim)
        self.b2 = PoseFiLMBlock(out_chs[1], out_chs[1], pose_dim=pose_dim, mode=mode)
        self.up3 = nn.ConvTranspose2d(out_chs[1], out_chs[2], 2, stride=2)
        self.g3 = PoseGate(skip_chs[2], pose_dim)
        self.b3 = PoseFiLMBlock(out_chs[2], out_chs[2], pose_dim=pose_dim, mode=mode)
        self.head = nn.Conv2d(out_chs[2], num_classes, 1)

    def forward(self, feat: torch.Tensor, skips: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                pose: torch.Tensor) -> torch.Tensor:
        x = self.up1(feat)
        x = self.g1(skips[0], x, pose)
        x = self.b1(x, pose)
        x = self.up2(x)
        x = self.g2(skips[1], x, pose)
        x = self.b2(x, pose)
        x = self.up3(x)
        x = self.g3(skips[2], x, pose)
        x = self.b3(x, pose)
        return self.head(x)


def l2m_build_decoder(backbone_out_ch: int,
                      skip_chs: Tuple[int, int, int],
                      out_chs: Tuple[int, int, int],
                      num_classes: int,
                      pose_dim: int,
                      mode: str = "film") -> L2MDecoderWithPose:
    return L2MDecoderWithPose(backbone_out_ch, skip_chs, out_chs, num_classes=num_classes, pose_dim=pose_dim, mode=mode)


def l2m_pose_forward(decoder: L2MDecoderWithPose,
                     backbone_feat: torch.Tensor,
                     skip_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     pose: torch.Tensor) -> torch.Tensor:
    return decoder(backbone_feat, skip_feats, pose)


def l2m_active_reward(prev_dist: float, curr_dist: float, prev_ent: float, curr_ent: float,
                      alpha: float = 0.4, clip: float = 1.0) -> float:
    return compute_active_reward(prev_dist, curr_dist, prev_ent, curr_ent, alpha=alpha, clip=clip)


def l2m_batch_reward(prev_d: torch.Tensor, curr_d: torch.Tensor, prev_e: torch.Tensor, curr_e: torch.Tensor,
                     alpha: float = 0.4, clip: float = 1.0) -> torch.Tensor:
    return batch_active_reward(prev_d, curr_d, prev_e, curr_e, alpha=alpha, clip=clip)


def l2m_consistency(student_logits: torch.Tensor, teacher_prob: torch.Tensor, valid_mask: Optional[torch.Tensor],
                    w: float = 0.3, mode: str = "l1", temp: float = 1.0) -> torch.Tensor:
    return consistency_loss(student_logits, teacher_prob, valid_mask, w=w, mode=mode,
                            student_is_logit=True, teacher_is_logit=False, temperature=temp)


class L2MIntegrationHelper:
    def __init__(self, num_classes: int = 21, pose_dim: int = 3,
                 skip_chs: Tuple[int, int, int] = (128, 64, 32),
                 out_chs: Tuple[int, int, int] = (128, 64, 32),
                 kd_weight: float = 0.3, kd_mode: str = "l1", alpha: float = 0.4):
        self.num_classes = num_classes
        self.pose_dim = pose_dim
        self.skip_chs = skip_chs
        self.out_chs = out_chs
        self.kd_weight = kd_weight
        self.kd_mode = kd_mode
        self.alpha = alpha
        self.decoder: Optional[L2MDecoderWithPose] = None
        self.metrics_logger = MetricsLogger()

    def build(self, backbone_out_ch: int, device: Optional[torch.device] = None) -> L2MDecoderWithPose:
        self.decoder = l2m_build_decoder(backbone_out_ch, self.skip_chs, self.out_chs,
                                         num_classes=self.num_classes, pose_dim=self.pose_dim, mode="film")
        if device is not None:
            self.decoder.to(device)
        return self.decoder

    def loss_with_kd(self, logits: torch.Tensor, target: torch.Tensor,
                     teacher_prob: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, ignore_index=-1)
        if teacher_prob is None:
            return ce
        kd = l2m_consistency(logits, teacher_prob, valid_mask, w=self.kd_weight, mode=self.kd_mode)
        return ce + kd

    def step_reward(self, prev_d: float, curr_d: float, prev_ent: float, curr_ent: float, clip: float = 1.0) -> float:
        return l2m_active_reward(prev_d, curr_d, prev_ent, curr_ent, alpha=self.alpha, clip=clip)

    def log_nav_metrics(self, path_lengths: List[float], shortest: List[float], success_flags: List[int],
                        progress_rates: Optional[List[float]] = None, times_to_success: Optional[List[float]] = None) -> Dict[str, float]:
        s = success_rate(success_flags)
        spl = spl_score(path_lengths, shortest, success_flags)
        soft = soft_spl_score(path_lengths, shortest, progress_rates if progress_rates is not None else [0.0] * len(path_lengths))
        dts = dts_score(times_to_success if times_to_success is not None else [t if t > 0 else 1.0 for t in range(1, len(path_lengths) + 1)])
        out = {"Success": s, "SPL": spl, "SoftSPL": soft, "DTS": dts}
        self.metrics_logger.log_row(out)
        return out

    def log_map_metrics(self, pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> Dict[str, float]:
        agg = MapMetricsAggregator(num_classes)
        agg.update(pred, gt)
        res = agg.compute()
        self.metrics_logger.log_row(res)
        return res

    def save_metrics(self, path: str):
        self.metrics_logger.save_csv(path)
