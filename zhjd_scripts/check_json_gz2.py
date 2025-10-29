import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import habitat
from habitat.config.default import get_config
import habitat.utils.visualizations.maps as map_util
from datasets.util import map_utils
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
from models.img_segmentation import get_img_segmentor_from_options
import test_utils as tutils

import torchvision.transforms as transforms

import gzip
import json


# ----------------------------
# 数据集类
# ----------------------------
class ObjNavEpisodeDataset(Dataset):
    def __init__(self, episode_files):
        self.episodes_file_list = episode_files

    def __len__(self):
        return len(self.episodes_file_list)

    def __getitem__(self, idx):
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        abs_pose = ep['abs_pose'][-10:]
        ego_grid_crops_spatial = torch.from_numpy(ep['ego_grid_crops_spatial'][-10:])
        step_ego_grid_crops_spatial = torch.from_numpy(ep['step_ego_grid_crops_spatial'][-10:])
        gt_grid_crops_spatial = torch.from_numpy(ep['gt_grid_crops_spatial'][-10:])
        gt_grid_crops_objects = torch.from_numpy(ep['gt_grid_crops_objects'][-10:])

        # 计算相对位姿
        rel_pose = []
        for i in range(abs_pose.shape[0]):
            rel_pose.append(utils.get_rel_pose(pos2=abs_pose[i], pos1=abs_pose[0]))

        item = {
            'pose': torch.from_numpy(np.asarray(rel_pose)).float(),
            'abs_pose': torch.from_numpy(abs_pose).float(),
            'ego_grid_crops_spatial': ego_grid_crops_spatial,
            'step_ego_grid_crops_spatial': step_ego_grid_crops_spatial,
            'gt_grid_crops_spatial': gt_grid_crops_spatial,
            'gt_grid_crops_objects': gt_grid_crops_objects,

            'images': torch.from_numpy(ep['images'][-10:]),
            'gt_segm': torch.from_numpy(ep['ssegs'][-10:]).type(torch.int64),
            'depth_imgs': torch.from_numpy(ep['depth_imgs'][-10:]),

            'pred_ego_crops_sseg': torch.from_numpy(ep['pred_ego_crops_sseg'][-10:]),
            'step_ego_grid_27': torch.from_numpy(ep['step_ego_grid_27'][-10:])
        }

        return item

def normalize_rgb(rgb_tensor):
    """
    将 RGB float tensor 转换为 uint8 显示图像
    支持范围为 [0, 1] 或 [-1, 1]
    """
    rgb_np = rgb_tensor.detach().cpu().numpy()
    if rgb_np.dtype in [np.float32, np.float64]:
        if rgb_np.max() <= 1.0 and rgb_np.min() >= 0.0:
            rgb_np = rgb_np * 255.0
        elif rgb_np.min() >= -1.0 and rgb_np.max() <= 1.0:
            rgb_np = (rgb_np + 1.0) * 127.5
    rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
    return rgb_np

def tensor_to_np(t):
    """确保 Tensor 是 numpy 格式，且 squeeze 掉 batch/channel 维度"""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        return t.numpy()
    return t

def inspect_npz_file(npz_path):
    if not os.path.exists(npz_path):
        print(f"❌ 文件未找到: {npz_path}")
        return

    print(f"📂 Inspecting: {npz_path}\n")

    with np.load(npz_path) as data:
        for key in data.files:
            value = data[key]
            print(f"🔑 {key:<30} | shape: {value.shape!s:<20} | dtype: {value.dtype}")


def visualize_all_fields(item, timestep=0):
    """
    显示所有字段的名字
    """


    # print("egocentric_spatial_grid_map shape:", egocentric_spatial_grid_map.shape)
    # print("                            dtype:", egocentric_spatial_grid_map.dtype)
    # print("step_grid_27 shape:", step_grid_27.shape)
    # print("                       dtype:", step_grid_27.dtype)



# ----------------------------
# 主函数入口
# ----------------------------
if __name__ == "__main__":
    root_path = "/home/robotlab/dataset/Test_Episodes/easy/v3/test/content"
    file_path = root_path + '/' + '2azQ1b91cZZ.json.gz'

    if not os.path.exists(file_path):
        print(f"❌ 文件未找到: {file_path}")
        exit(1)

    dataset = ObjNavEpisodeDataset([file_path])
    item = dataset[0]

    # for t in range(4):
    #     print(f"\n=== 可视化时间步 {t} ===")
    #     visualize_item(item, timestep=t)

    for t in range(10):
        print(f"🕒 时间步 {t}")
        visualize_all_fields(item, timestep=t)