import math
from dataclasses import dataclass
# zhjd
# from typing import Tuple, Optional, Union, Literal
# from typing_extensions import Tuple, Optional, Union, Literal
from typing import Tuple, Optional, Union
from typing_extensions import Literal

from habitat.core.simulator import Sensor, SensorTypes
from torch import Tensor

import numpy as np
import torch
import torch.nn.functional as F


Tensor = torch.Tensor
ArrayLike = Union[np.ndarray, Tensor]


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
        K = torch.tensor([[self.fx, 0.0, self.cx],
                          [0.0, self.fy, self.cy],
                          [0.0, 0.0, 1.0]], dtype=dtype, device=device)
        return K

    def scale(self, sx: float, sy: Optional[float] = None) -> "CameraIntrinsics":
        if sy is None:
            sy = sx
        fx = self.fx * sx
        fy = self.fy * sy
        cx = self.cx * sx
        cy = self.cy * sy
        w = int(round(self.width * sx))
        h = int(round(self.height * sy))
        return CameraIntrinsics(fx, fy, cx, cy, w, h)

    def to_dict(self) -> dict:
        return {"fx": float(self.fx), "fy": float(self.fy), "cx": float(self.cx), "cy": float(self.cy),
                "width": int(self.width), "height": int(self.height)}


# ---------- 基础工具 ----------

def to_homogeneous(x: Tensor) -> Tensor:
    ones = torch.ones((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=-1)


def from_homogeneous(x: Tensor, eps: float = 1e-8) -> Tensor:
    w = x[..., -1:].clamp(min=eps)
    return x[..., :-1] / w


def normalize_image_coords(u: Tensor, v: Tensor, width: int, height: int) -> Tuple[Tensor, Tensor]:
    xn = 2.0 * (u / max(width - 1, 1)) - 1.0
    yn = 2.0 * (v / max(height - 1, 1)) - 1.0
    return xn, yn


def denormalize_image_coords(xn: Tensor, yn: Tensor, width: int, height: int) -> Tuple[Tensor, Tensor]:
    u = (xn + 1.0) * 0.5 * (width - 1)
    v = (yn + 1.0) * 0.5 * (height - 1)
    return u, v


def pack_intrinsics(fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(fx, fy, cx, cy, width, height)


def intrinsics_from_matrix(K: ArrayLike, width: int, height: int) -> CameraIntrinsics:
    if isinstance(K, np.ndarray):
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    else:
        fx, fy, cx, cy = float(K[0, 0].item()), float(K[1, 1].item()), float(K[0, 2].item()), float(K[1, 2].item())
    return CameraIntrinsics(fx, fy, cx, cy, int(width), int(height))


# ---------- 等比 pad 到方形（letterbox） ----------

def letterbox_pad_torch(img: Tensor, pad_value: float = 0.0) -> Tuple[Tensor, Tuple[int, int]]:
    assert img.dim() == 4, "img: [B,C,H,W]"
    B, C, H, W = img.shape
    side = max(H, W)
    out = img.new_full((B, C, side, side), pad_value)
    ph = (side - H) // 2
    pw = (side - W) // 2
    out[:, :, ph:ph + H, pw:pw + W] = img
    return out, (ph, pw)


def letterbox_unpad_coords(u: Tensor, v: Tensor, pad_hw: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
    ph, pw = pad_hw
    return u - pw, v - ph


def numpy_letterbox_pad(img: np.ndarray, pad_value: float = 0.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    assert img.ndim in (3, 4)
    if img.ndim == 3:
        H, W, C = img.shape
        side = max(H, W)
        out = np.full((side, side, C), pad_value, dtype=img.dtype)
        ph = (side - H) // 2
        pw = (side - W) // 2
        out[ph:ph + H, pw:pw + W] = img
        return out, (ph, pw)
    else:
        N, H, W, C = img.shape
        side = max(H, W)
        out = np.full((N, side, side, C), pad_value, dtype=img.dtype)
        ph = (side - H) // 2
        pw = (side - W) // 2
        out[:, ph:ph + H, pw:pw + W] = img
        return out, (ph, pw)


# ---------- 投影/反投影 ----------

def meshgrid_xy(width: int, height: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tuple[Tensor, Tensor]:
    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    v, u = torch.meshgrid(ys, xs, indexing="ij")
    return u, v


def unproject_depth_to_points(depth: Tensor, intr: CameraIntrinsics) -> Tensor:
    assert depth.dim() == 4 and depth.shape[1] == 1
    print(f"[DEBUG] depth.shape = {depth.shape}")   # [DEBUG] pts.shape = torch.Size([1, 1, 3, 240, 320])
    B, _, H, W = depth.shape
    u, v = meshgrid_xy(W, H, device=depth.device, dtype=depth.dtype)
    u = u[None, None].expand(B, 1, H, W)
    v = v[None, None].expand(B, 1, H, W)
    z = depth.clamp_min(1e-6)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    # zhjd
    # pts = torch.stack([x, y, z], dim=2)  # [B,3,H,W]
    pts = torch.stack([x, y, z], dim=2)  # [B,3,H,W]
    pts = pts.squeeze(1)
    pts = pts.permute(0, 2, 3, 1).reshape(B, -1, 3)
    return pts


def project_points_to_image(pts_cam: Tensor, intr: CameraIntrinsics) -> Tuple[Tensor, Tensor]:
    x, y, z = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2].clamp_min(1e-8)
    u = intr.fx * x / z + intr.cx
    v = intr.fy * y / z + intr.cy
    return u, v


# ---------- SE(2) 与 SE(3) ----------

def se2_from_xytheta(x: Tensor, y: Tensor, theta: Tensor) -> Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    one = torch.ones_like(c)
    T = torch.stack([
        torch.stack([c, -s, x], dim=-1),
        torch.stack([s,  c, y], dim=-1),
        torch.stack([z,  z, one], dim=-1),
    ], dim=-2)
    return T


def se2_inverse(T: Tensor) -> Tensor:
    R = T[..., :2, :2]
    t = T[..., :2, 2:]
    Rt = R.transpose(-1, -2)
    ti = -Rt @ t
    out = T.clone()
    out[..., :2, :2] = Rt
    out[..., :2, 2:] = ti
    out[..., 2, :] = torch.tensor([0.0, 0.0, 1.0], dtype=out.dtype, device=out.device)
    return out


def se2_compose(A: Tensor, B: Tensor) -> Tensor:
    C = torch.empty_like(A)
    C[..., :2, :2] = A[..., :2, :2] @ B[..., :2, :2]
    C[..., :2, 2:] = A[..., :2, :2] @ B[..., :2, 2:] + A[..., :2, 2:]
    C[..., 2, :] = torch.tensor([0.0, 0.0, 1.0], dtype=A.dtype, device=A.device)
    return C


def se2_apply(T: Tensor, pts2: Tensor) -> Tensor:
    if pts2.shape[-1] == 2:
        pts2 = to_homogeneous(pts2)
    out = (T @ pts2.unsqueeze(-1)).squeeze(-1)
    return from_homogeneous(out)


def quat_normalize(q: Tensor, eps: float = 1e-8) -> Tensor:
    n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / n


def quat_to_matrix(q: Tensor) -> Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    ww = w * w; xx = x * x; yy = y * y; zz = z * z
    wx = w * x; wy = w * y; wz = w * z
    xy = x * y; xz = x * z; yz = y * z
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)
    R = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)
    return R


def se3_from_rt(R: Tensor, t: Tensor) -> Tensor:
    B = R.shape[:-2]
    T = torch.zeros((*B, 4, 4), dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_from_quat_t(quat: Tensor, t: Tensor) -> Tensor:
    R = quat_to_matrix(quat)
    return se3_from_rt(R, t)


def se3_inverse(T: Tensor) -> Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    ti = -Rt @ t
    Ti = torch.zeros_like(T)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3:] = ti
    Ti[..., 3, 3] = 1.0
    return Ti


def se3_compose(A: Tensor, B: Tensor) -> Tensor:
    C = torch.zeros_like(A)
    C[..., :3, :3] = A[..., :3, :3] @ B[..., :3, :3]
    C[..., :3, 3] = (A[..., :3, :3] @ B[..., :3, 3:].unsqueeze(-1)).squeeze(-1) + A[..., :3, 3]
    C[..., 3, 3] = 1.0
    return C


def se3_apply(T: Tensor, pts3: Tensor) -> Tensor:
    if pts3.shape[-1] == 3:
        pts3 = to_homogeneous(pts3)
    out = (T @ pts3.unsqueeze(-1)).squeeze(-1)
    return from_homogeneous(out)


# ---------- 相机坐标系转换 ----------

def world_to_camera(pts_world: Tensor, T_cam_world: Tensor) -> Tensor:
    T = se3_inverse(T_cam_world)
    return se3_apply(T, pts_world)


def camera_to_world(pts_cam: Tensor, T_cam_world: Tensor) -> Tensor:
    return se3_apply(T_cam_world, pts_cam)


# ---------- 深度/点云/射线 ----------

def depth_to_pointcloud(depth: Tensor, intr: CameraIntrinsics, T_cam_world: Optional[Tensor] = None, valid_min: float = 1e-6) -> Tensor:
    pts_cam = unproject_depth_to_points(depth, intr)
    if T_cam_world is None:
        return pts_cam
    return camera_to_world(pts_cam, T_cam_world)


def make_rays(intr: CameraIntrinsics, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tuple[Tensor, Tensor]:
    H, W = intr.height, intr.width
    u, v = meshgrid_xy(W, H, device=device, dtype=dtype)
    x = (u - intr.cx) / intr.fx
    y = (v - intr.cy) / intr.fy
    z = torch.ones_like(x)
    dirs = torch.stack([x, y, z], dim=-1)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-9)
    origins = torch.zeros_like(dirs)
    return origins, dirs


def frustum_corners(depth_min: float, depth_max: float, intr: CameraIntrinsics, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> Tensor:
    H, W = intr.height, intr.width
    xs = torch.tensor([0.0, W - 1.0], device=device, dtype=dtype)
    ys = torch.tensor([0.0, H - 1.0], device=device, dtype=dtype)
    uu, vv = torch.meshgrid(ys, xs, indexing="ij")
    z_near = torch.full_like(uu, depth_min)
    z_far = torch.full_like(uu, depth_max)
    x_n = (uu - intr.cx) * z_near / intr.fx
    y_n = (vv - intr.cy) * z_near / intr.fy
    x_f = (uu - intr.cx) * z_far / intr.fx
    y_f = (vv - intr.cy) * z_far / intr.fy
    near = torch.stack([x_n, y_n, z_near], dim=-1).reshape(-1, 3)
    far = torch.stack([x_f, y_f, z_far], dim=-1).reshape(-1, 3)
    return torch.cat([near, far], dim=0)


# ---------- 采样/重映射 ----------

def grid_sample_bilinear(feat: Tensor, uv: Tensor, align_corners: bool = False) -> Tensor:
    B, C, H, W = feat.shape
    u, v = uv[..., 0], uv[..., 1]
    xn, yn = normalize_image_coords(u, v, W, H)
    grid = torch.stack([xn, yn], dim=-1).to(feat.dtype)
    grid = grid.view(B, -1, 1, 2)
    out = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    return out.view(B, C, -1)


# ---------- BEV 网格与占据 ----------

def xy_to_bev_indices(xy: Tensor, voxel_size: float, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Tuple[Tensor, Tensor]:
    x0, x1 = x_range
    y0, y1 = y_range
    ix = torch.floor((xy[..., 0] - x0) / voxel_size).long()
    iy = torch.floor((xy[..., 1] - y0) / voxel_size).long()
    W = int(math.ceil((x1 - x0) / voxel_size))
    H = int(math.ceil((y1 - y0) / voxel_size))
    mask = (ix >= 0) & (iy >= 0) & (ix < W) & (iy < H)
    idx = torch.stack([iy.clamp_min(0), ix.clamp_min(0)], dim=-1)
    return idx, mask


def scatter_bev_occupancy(xy: Tensor, grid: Tensor, voxel_size: float, x_range: Tuple[float, float], y_range: Tuple[float, float], value: float = 1.0):
    idx, m = xy_to_bev_indices(xy, voxel_size, x_range, y_range)
    iy, ix = idx[m].unbind(-1)
    grid[iy, ix] = value


def update_log_odds(grid_logit: Tensor, idx: Tensor, mask: Tensor, hit_logit: float = 2.0, miss_logit: float = -0.5, clamp: Tuple[float, float] = (-4.0, 4.0)) -> Tensor:
    gy, gx = idx[..., 0], idx[..., 1]
    m = mask
    grid_logit[gy[m], gx[m]] = (grid_logit[gy[m], gx[m]] + hit_logit).clamp(*clamp)
    nm = (~m) & (gy >= 0) & (gx >= 0) & (gy < grid_logit.shape[0]) & (gx < grid_logit.shape[1])
    grid_logit[gy[nm], gx[nm]] = (grid_logit[gy[nm], gx[nm]] + miss_logit).clamp(*clamp)
    return grid_logit


# ---------- 重设尺寸并同步内参 ----------

def resize_image_with_intrinsics(img: Tensor, intr: CameraIntrinsics, out_wh: Tuple[int, int], mode: str = "bilinear") -> Tuple[Tensor, CameraIntrinsics]:
    assert img.dim() == 4
    B, C, H, W = img.shape
    ow, oh = out_wh
    sx = ow / max(W, 1)
    sy = oh / max(H, 1)
    out = F.interpolate(img, size=(oh, ow), mode=mode, align_corners=False if mode == "bilinear" else None)
    intr2 = intr.scale(sx, sy)
    intr2.width, intr2.height = ow, oh
    return out, intr2


# ---------- 深度/法线/重投影辅助 ----------

def backproject(uv: Tensor, depth: Tensor, intr: CameraIntrinsics) -> Tensor:
    u = uv[..., 0]; v = uv[..., 1]
    z = depth
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return torch.stack([x, y, z], dim=-1)


def reproject(pts_cam: Tensor, intr: CameraIntrinsics) -> Tensor:
    u, v = project_points_to_image(pts_cam, intr)
    return torch.stack([u, v], dim=-1)


def compute_normals_from_depth(depth: Tensor, intr: CameraIntrinsics) -> Tensor:
    B, _, H, W = depth.shape
    u, v = meshgrid_xy(W, H, device=depth.device, dtype=depth.dtype)
    du = torch.tensor([[1, -1]], device=depth.device, dtype=depth.dtype).view(1, 1, 1, 2)
    dv = torch.tensor([[1], [-1]], device=depth.device, dtype=depth.dtype).view(1, 1, 2, 1)
    dz_du = F.pad(depth, (1, 0, 0, 0)) - F.pad(depth, (0, 1, 0, 0))
    dz_dv = F.pad(depth, (0, 0, 1, 0)) - F.pad(depth, (0, 0, 0, 1))
    xu = (u[None, None] - intr.cx) / intr.fx
    yv = (v[None, None] - intr.cy) / intr.fy
    nx = -dz_du * xu
    ny = -dz_dv * yv
    nz = torch.ones_like(nx)
    n = torch.cat([nx, ny, nz], dim=1)
    n = n / torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(1e-9)
    return n


# ---------- 视锥/裁剪 ----------

def in_image_bounds(u: Tensor, v: Tensor, width: int, height: int) -> Tensor:
    return (u >= 0) & (v >= 0) & (u < width) & (v < height)


def depth_valid_mask(depth: Tensor, min_z: float = 1e-6, max_z: float = 1e6) -> Tensor:
    return (depth > min_z) & (depth < max_z)


# ---------- 统一接口 ----------

def pack_intrinsics_matrix(K: Tensor, width: int, height: int) -> CameraIntrinsics:
    return intrinsics_from_matrix(K, width, height)

# zhjd
def crop_and_update_intrinsics(img: Tensor, intr: CameraIntrinsics, crop_xywh: Tuple[int, int, int, int]) -> Tuple[Tensor, CameraIntrinsics]:
    x0, y0, w, h = crop_xywh
    out = img[..., y0:y0 + h, x0:x0 + w]
    intr2 = CameraIntrinsics(intr.fx, intr.fy, intr.cx - x0, intr.cy - y0, w, h)
    return out, intr2

# def crop_and_update_intrinsics(img: Tensor, intr: CameraIntrinsics, crop_xywh: Tuple[int, int, int, int]) -> Tuple[TENSOR, CameraIntrinsics]:
#     x0, y0, w, h = crop_xywh
#     out = img[..., y0:y0 + h, x0:x0 + w]
#     intr2 = CameraIntrinsics(intr.fx, intr.fy, intr.cx - x0, intr.cy - y0, w, h)
#     return out, intr2


def compose_se3_from_euler_t(rx: float, ry: float, rz: float, t: Tuple[float, float, float], device=None, dtype=torch.float32) -> Tensor:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = torch.tensor([[cz, -sz, 0.0],
                       [sz,  cz, 0.0],
                       [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    Ry = torch.tensor([[cy, 0.0, sy],
                       [0.0, 1.0, 0.0],
                       [-sy, 0.0, cy]], dtype=dtype, device=device)
    Rx = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, cx, -sx],
                       [0.0, sx,  cx]], dtype=dtype, device=device)
    R = Rz @ Ry @ Rx
    T = se3_from_rt(R, torch.tensor(t, dtype=dtype, device=device))
    return T


# ---------- 数值稳定/夹取 ----------

def safe_div(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    return a / b.clamp_min(eps)


def clamp_coords(u: Tensor, v: Tensor, width: int, height: int) -> Tuple[Tensor, Tensor]:
    return u.clamp(0, width - 1), v.clamp(0, height - 1)


# ---------- 快速验证 ----------

def _quick_sanity():
    intr = CameraIntrinsics(500.0, 500.0, 160.0, 120.0, 320, 240)
    depth = torch.ones(1, 1, 240, 320)
    pts = unproject_depth_to_points(depth, intr)
    u, v = project_points_to_image(pts, intr)
    assert (u.view(-1)[:10].isfinite().all() and v.view(-1)[:10].isfinite().all())


# 确保导入即能快速检查（不抛异常）
try:
    _quick_sanity()
except Exception:
    pass
