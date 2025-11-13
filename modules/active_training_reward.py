from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def map_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = prob.clamp(eps, 1 - eps)
    ent = -(p * p.log()).sum(dim=1)  # [B,H,W]
    return ent.mean(dim=(1, 2))      # [B]


def path_delta(prev_dist: torch.Tensor, curr_dist: torch.Tensor) -> torch.Tensor:
    return (prev_dist - curr_dist)


def compute_active_reward(
    prev_dist: float,
    curr_dist: float,
    prev_entropy: float,
    curr_entropy: float,
    alpha: float = 0.5,
    clip: Optional[float] = 1.0,
) -> float:
    dd = float(prev_dist - curr_dist)
    de = float(prev_entropy - curr_entropy)
    r = dd + alpha * de
    if clip is not None:
        r = max(min(r, clip), -clip)
    return r


class ActiveReward:
    def __init__(
        self,
        alpha: float = 0.5,
        clip: float = 1.0,
        dist_scale: float = 1.0,
        ent_scale: float = 1.0,
        ema: float = 0.0,
    ):
        self.alpha = alpha
        self.clip = clip
        self.dist_scale = dist_scale
        self.ent_scale = ent_scale
        self.ema = ema
        self._state: Optional[Tuple[float, float]] = None

    def reset(self):
        self._state = None

    def _smooth(self, val: float, old: Optional[float]) -> float:
        if old is None or self.ema <= 0.0:
            return val
        return (1.0 - self.ema) * val + self.ema * old

    def step(
        self,
        prev_dist: float,
        curr_dist: float,
        prev_entropy: float,
        curr_entropy: float,
    ) -> float:
        pd = self._smooth(prev_dist, self._state[0] if self._state else None)
        pe = self._smooth(prev_entropy, self._state[1] if self._state else None)
        self._state = (curr_dist, curr_entropy)
        dd = (pd - curr_dist) * self.dist_scale
        de = (pe - curr_entropy) * self.ent_scale
        r = dd + self.alpha * de
        if self.clip is not None:
            r = max(min(float(r), self.clip), -self.clip)
        return float(r)


def curriculum_alpha(step: int, warmup: int, max_alpha: float) -> float:
    if step <= 0:
        return 0.0
    if step >= warmup:
        return max_alpha
    return max_alpha * (step / float(warmup))


def batch_active_reward(
    prev_d: torch.Tensor,
    curr_d: torch.Tensor,
    prev_ent: torch.Tensor,
    curr_ent: torch.Tensor,
    alpha: float = 0.5,
    clip: Optional[float] = 1.0,
) -> torch.Tensor:
    dd = (prev_d - curr_d)
    de = (prev_ent - curr_ent)
    r = dd + alpha * de
    if clip is not None:
        r = torch.clamp(r, -clip, clip)
    return r
