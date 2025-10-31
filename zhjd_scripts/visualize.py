#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在窗口中直接动态展示 NPZ 文件中的 ego_grid_crops_spatial 等数据；
不会保存任何文件到本地。

Example:
    python visualize_npz_grid_show.py --file ep_1_1_2azQ1b91cZZ.npz --key ego_grid_crops_spatial --fps 2
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------------
# 颜色映射定义
# ----------------------------
color_mapping_3 = {
    0: (200, 200, 200),  # unknown
    1: (0, 0, 0),        # obstacle
    2: (255, 255, 255)   # free
}

# 为 color_mapping_27 随机生成颜色（可按需固定）
color_mapping_27 = {i: tuple(np.random.randint(0, 255, 3)) for i in range(27)}
color_mapping_27[0] = (0, 0, 0)  # 背景为黑色

# ----------------------------
# 上色函数（模仿 Habitat 的 colorize_grid）
# ----------------------------
def colorize_grid(grid, color_mapping=27):
    """输入: grid shape = B x T x C x H x W"""
    if isinstance(grid, np.ndarray):
        grid = torch.tensor(grid)
    grid = grid.detach().cpu()
    grid_img = torch.zeros((*grid.shape[:2], grid.shape[3], grid.shape[4], 3), dtype=torch.uint8)

    if grid.shape[2] > 1:
        grid_prob_max = torch.amax(grid, dim=2)
        inds = (grid_prob_max <= 0.05).nonzero()
        if len(inds) > 0:
            grid[inds[:, 0], inds[:, 1], 0, inds[:, 2], inds[:, 3]] = 1
        grid = torch.argmax(grid, dim=2)
    else:
        grid = grid.squeeze(2)

    color_mapping_dict = color_mapping_27 if color_mapping == 27 else color_mapping_3
    for label, color in color_mapping_dict.items():
        mask = (grid == label)
        grid_img[mask] = torch.tensor(color, dtype=torch.uint8)

    return grid_img.permute(0, 1, 4, 2, 3)  # B x T x 3 x H x W


# ----------------------------
# 主函数
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize NPZ grids in a window (no file saved).")
    parser.add_argument("--file", required=True, help="Path to the NPZ file")
    parser.add_argument("--key", required=True, help="Key inside NPZ to visualize")
    parser.add_argument("--fps", type=float, default=1, help="Frames per second for animation")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ 文件未找到: {args.file}")
        exit(1)

    data = np.load(args.file)
    if args.key not in data.files:
        print(f"❌ 键 '{args.key}' 不存在，文件中包含: {data.files}")
        exit(1)

    ego_grid = data[args.key]
    print(f"✅ 读取键 '{args.key}', shape={ego_grid.shape}")

    # 若是 (T, C, H, W)，则补上 batch 维度
    if ego_grid.ndim == 4:
        ego_grid = ego_grid[None, ...]

    # 上色
    colored = colorize_grid(torch.tensor(ego_grid), color_mapping=3)
    colored_np = colored[0].numpy().transpose(0, 2, 3, 1)  # T x H x W x 3

    # ----------------------------
    # 动态展示
    # ----------------------------
    fig, ax = plt.subplots()
    ax.axis("off")
    img = ax.imshow(colored_np[0])

    def update(frame):
        img.set_data(colored_np[frame])
        ax.set_title(f"{args.key} (frame {frame+1}/{colored_np.shape[0]})")
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=range(colored_np.shape[0]),
                                  interval=1000/args.fps, blit=True, repeat=True)

    print(f"🎥 播放动画（fps={args.fps}）... 关闭窗口以退出。")
    plt.show()


# ----------------------------
# 入口
# ----------------------------
if __name__ == "__main__":
    main()
