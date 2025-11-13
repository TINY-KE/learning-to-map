import os
import csv
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def success_rate(success_flags: List[int]) -> float:
    if len(success_flags) == 0:
        return 0.0
    return float(np.mean(np.array(success_flags, dtype=np.float32)))


def spl_score(path_lengths: List[float], shortest_paths: List[float], success_flags: List[int]) -> float:
    n = len(path_lengths)
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        if success_flags[i]:
            sp = max(shortest_paths[i], 1e-6)
            s += shortest_paths[i] / max(path_lengths[i], sp)
    return float(s / n)


def soft_spl_score(path_lengths: List[float], shortest_paths: List[float], progress_rates: List[float]) -> float:
    n = len(path_lengths)
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        pr = max(progress_rates[i], 0.0)
        denom = max(path_lengths[i], shortest_paths[i], 1e-6)
        s += pr * shortest_paths[i] / denom
    return float(s / n)


def dts_score(time_to_success_list: List[float]) -> float:
    if len(time_to_success_list) == 0:
        return 0.0
    arr = np.array(time_to_success_list, dtype=np.float64)
    arr = np.maximum(arr, 1e-6)
    return float(np.mean(1.0 / arr))


def confusion_matrix(pred: torch.Tensor, gt: torch.Tensor, num_classes: int, ignore_index: int = -1) -> torch.Tensor:
    with torch.no_grad():
        p = pred.argmax(dim=1).view(-1)
        g = gt.view(-1)
        mask = g != ignore_index
        p = p[mask]
        g = g[mask]
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred.device)
        idx = num_classes * g + p
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        cm += binc.view(num_classes, num_classes)
        return cm


def iou_f1_acc_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        tp = torch.diag(cm).to(torch.float32)
        fp = cm.sum(dim=0).to(torch.float32) - tp
        fn = cm.sum(dim=1).to(torch.float32) - tp
        denom_iou = (tp + fp + fn).clamp_min(1e-6)
        iou = tp / denom_iou
        prec = tp / (tp + fp).clamp_min(1e-6)
        rec = tp / (tp + fn).clamp_min(1e-6)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-6)
        acc = tp.sum() / cm.sum().clamp_min(1e-6)
        miou = iou.mean().item()
        mf1 = f1.mean().item()
        return {
            "mIoU": float(miou),
            "mF1": float(mf1),
            "Acc": float(acc.item()),
        }


class MapMetricsAggregator:
    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        cm = confusion_matrix(pred, gt, self.num_classes, self.ignore_index).cpu()
        self.cm += cm

    def compute(self) -> Dict[str, float]:
        return iou_f1_acc_from_cm(self.cm)


class MetricsLogger:
    def __init__(self, headers: Optional[List[str]] = None):
        self.rows: List[Dict[str, float]] = []
        self.headers = headers

    def log_row(self, row: Dict[str, float]):
        self.rows.append(row)

    def save_csv(self, path: str):
        if len(self.rows) == 0:
            return
        keys = self.headers if self.headers is not None else sorted({k for r in self.rows for k in r.keys()})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in self.rows:
                w.writerow([r.get(k, "") for k in keys])
