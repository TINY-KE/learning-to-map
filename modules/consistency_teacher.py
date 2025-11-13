# from typing_extensions import Optional, Literal, Tuple
# zhjd
from typing import Tuple, Optional, Union
from typing_extensions import Literal


import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_prob(x: torch.Tensor, is_logit: bool, temp: float) -> torch.Tensor:
    if is_logit:
        if temp != 1.0:
            x = x / temp
        return torch.softmax(x, dim=1)
    return x


def _resize_like(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if a.shape[2:] == ref.shape[2:]:
        return a
    return F.interpolate(a, size=ref.shape[2:], mode="bilinear", align_corners=False)


def entropy_2d(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = prob.clamp(eps, 1 - eps)
    ent = -(p * p.log()).sum(dim=1)
    return ent.mean(dim=(1, 2))


def consistency_loss(
    student_map: torch.Tensor,
    teacher_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    w: float = 0.3,
    mode: Literal["l1", "l2", "kl", "js"] = "l1",
    student_is_logit: bool = True,
    teacher_is_logit: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    s = _ensure_prob(student_map, student_is_logit, temperature)
    t = _ensure_prob(teacher_map, teacher_is_logit, 1.0)
    t = _resize_like(t, s)
    if mode == "l1":
        diff = (s - t).abs().mean(dim=1, keepdim=True)
    elif mode == "l2":
        diff = (s - t).pow(2).mean(dim=1, keepdim=True)
    elif mode == "kl":
        s = s.clamp(1e-6, 1 - 1e-6)
        t = t.clamp(1e-6, 1 - 1e-6)
        diff = (s * (torch.log(s) - torch.log(t))).sum(dim=1, keepdim=True)
    else:
        m = 0.5 * (s + t)
        s = s.clamp(1e-6, 1 - 1e-6)
        t = t.clamp(1e-6, 1 - 1e-6)
        kl_sm = (s * (torch.log(s) - torch.log(m))).sum(dim=1, keepdim=True)
        kl_tm = (t * (torch.log(t) - torch.log(m))).sum(dim=1, keepdim=True)
        diff = 0.5 * (kl_sm + kl_tm)
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = F.interpolate(mask.float(), size=diff.shape[2:], mode="nearest")
        loss = (diff * mask).sum() / (mask.sum() + 1e-6)
    else:
        loss = diff.mean()
    return w * loss


class TeacherAdaptor(nn.Module):
    def __init__(self, in_ch: int, out_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_classes, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.conv(x), dim=1)


def kd_with_entropy(
    student_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    kd_w: float = 0.3,
    kd_mode: str = "l1",
    temp: float = 1.0,
    ent_w: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_prob = _ensure_prob(student_logits, True, temp)
    kd = consistency_loss(
        s_prob, teacher_prob, valid_mask, w=kd_w, mode=kd_mode, student_is_logit=False, teacher_is_logit=False
    )
    ent = entropy_2d(s_prob).mean() * ent_w
    return kd + ent, s_prob.detach()
