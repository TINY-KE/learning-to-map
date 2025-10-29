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

class ObjNavEpisodeDataset(Dataset):
    def __init__(self, episode_files):
        self.episodes_file_list = episode_files

    def __len__(self):
        return len(self.episodes_file_list)

    def __getitem__(self, idx):
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        abs_pose = ep['abs_pose'][-10:]
        ego_grid_crops_spatial = torch.from_numpy(ep['r'][-10:])
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


def inspect_npz_file(npz_path):
    if not os.path.exists(npz_path):
        print(f"❌ 文件未找到: {npz_path}")
        return

    print(f"📂 Inspecting: {npz_path}\n")

    with np.load(npz_path) as data:
        for key in data.files:
            value = data[key]
            print(f"🔑 {key:<30} | shape: {value.shape!s:<20} | dtype: {value.dtype}")

# 示例用法
if __name__ == "__main__":
    root_path = "/home/robotlab/habitat-api/data/scene_datasets/mp3d/val/1"
    # ep_path = root_path + '/' + '1LXtFkjw3qL_color.npz'
    ep_path = root_path + '/' + '1LXtFkjw3qL_pcloud.npz'
    inspect_npz_file(ep_path)