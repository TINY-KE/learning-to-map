import math
# zhjd
# from typing_extensions import Optional, Tuple, Literal
from typing import Tuple, Optional, Union
from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNormalizer(nn.Module):
    def __init__(self, mode: Literal["xytheta", "xysin", "raw"] = "xytheta", eps: float = 1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if self.mode == "raw":
            return pose
        if self.mode == "xysin":
            if pose.shape[-1] == 3:
                x, y, th = pose.unbind(-1)
                c = torch.cos(th)
                s = torch.sin(th)
                return torch.stack([x, y, c, s], dim=-1)
            return pose
        # xytheta -> 归一化角度到 [-pi, pi]
        if pose.shape[-1] >= 3:
            out = pose.clone()
            out[..., 2] = (out[..., 2] + math.pi) % (2 * math.pi) - math.pi
            return out
        return pose


class PoseConditionLayer(nn.Module):
    def __init__(self, feat_dim: int, pose_dim: int = 3, mode: Literal["film", "adain"] = "film"):
        super().__init__()
        self.mode = mode
        self.gamma = nn.Linear(pose_dim, feat_dim)
        self.beta = nn.Linear(pose_dim, feat_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feat: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        if self.mode == "adain":
            mean = feat.mean((2, 3), keepdim=True)
            std = feat.std((2, 3), keepdim=True).clamp_min(1e-6)
            feat = (feat - mean) / std
        g = self.gamma(pose).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(pose).unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + g) + b


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PoseFiLMBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pose_dim: int = 3, mode: Literal["film", "adain"] = "film"):
        super().__init__()
        self.c1 = ConvBNAct(in_ch, out_ch)
        self.mod = PoseConditionLayer(out_ch, pose_dim, mode)
        self.c2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.mod(x, pose)
        x = self.c2(x)
        return x


class PoseGate(nn.Module):
    def __init__(self, skip_ch: int, pose_dim: int = 3):
        super().__init__()
        self.fc = nn.Linear(pose_dim, skip_ch)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.proj: Optional[nn.Conv2d] = None

    def forward(self, skip: torch.Tensor, main: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.fc(pose)).unsqueeze(-1).unsqueeze(-1)
        gated = skip * g
        if gated.shape[1] != main.shape[1]:
            if self.proj is None or self.proj.weight.shape[0] != main.shape[1] or self.proj.weight.shape[1] != gated.shape[1]:
                self.proj = nn.Conv2d(gated.shape[1], main.shape[1], 1, bias=False, device=main.device)
            gated = self.proj(gated)
        return main + gated


class PoseUNetDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, pose_dim: int = 3, mode: str = "film"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.gate = PoseGate(skip_ch if skip_ch > 0 else out_ch, pose_dim=pose_dim)
        self.film = PoseFiLMBlock(out_ch, out_ch, pose_dim=pose_dim, mode=mode)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor], pose: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = self.gate(skip, x, pose)
        x = self.film(x, pose)
        return x


class PoseAwareDecoder(nn.Module):
    def __init__(self, chs: Tuple[int, int, int, int] = (256, 128, 64, 32), num_classes: int = 21, pose_dim: int = 3, mode: str = "film"):
        super().__init__()
        self.blocks = nn.ModuleList([
            PoseUNetDecoderBlock(chs[0], chs[1], chs[1], pose_dim, mode),
            PoseUNetDecoderBlock(chs[1], chs[2], chs[2], pose_dim, mode),
            PoseUNetDecoderBlock(chs[2], chs[3], chs[3], pose_dim, mode),
        ])
        self.head = nn.Conv2d(chs[3], num_classes, 1)
        self.norm = PoseNormalizer("xytheta")

    def forward(self, feat: torch.Tensor, skips: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], pose: torch.Tensor) -> torch.Tensor:
        pose = self.norm(pose)
        x = feat
        x = self.blocks[0](x, skips[0], pose)
        x = self.blocks[1](x, skips[1], pose)
        x = self.blocks[2](x, skips[2], pose)
        return self.head(x)
