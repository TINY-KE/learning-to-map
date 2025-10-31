
import multiprocessing as mp
from multiprocessing import Pool, TimeoutError
import numpy as np
from datasets.dataloader import HabitatDataScene
import datasets.util.utils as utils
import os
import argparse
import torch
import random
import json

# 创建命令行参数解析器。
class Params(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # 数据划分：训练 / 验证 / 测试。
        # 会决定加载的 YAML 配置文件是哪一个。
        self.parser.add_argument('--split', type=str, dest='split', default='train',
                                 choices=['train', 'val', 'test'])

        self.parser.add_argument('--grid_dim', type=int, dest='grid_dim', default=384)
        self.parser.add_argument('--crop_size', type=int, dest='crop_size', default=64)
        self.parser.add_argument('--cell_size', type=float, dest='cell_size', default=0.1)
        self.parser.add_argument('--turn_angle', type=int, dest='turn_angle', default=30)
        self.parser.add_argument('--forward_step_size', type=float, dest='forward_step_size', default=0.25)

        # TODO: 目标物体类别数：27 类（ObjectNav 常见的 object set）。
        # todo: 空间类别数：3 类（通常是 free / obstacle / unknown）。
        self.parser.add_argument('--n_object_classes', type=int, dest='n_object_classes', default=27)
        self.parser.add_argument('--n_spatial_classes', type=int, dest='n_spatial_classes', default=3)

        self.parser.add_argument('--img_size', dest='img_size', type=int, default=256)
        self.parser.add_argument('--img_segm_size', dest='img_segm_size', type=int, default=128)

        # 每个 Scene 最多生成多少个 episode 文件。
        self.parser.add_argument('--max_num_episodes', dest='max_num_episodes', type=int, default=2500)

        # 每个 episode 包含多少个时间步（trajectory 长度）。
        # truncate_ep=True → 截取最后 N 步，不跑完整个 episode。
        self.parser.add_argument('--episode_len', type=int, dest='episode_len', default=10)
        self.parser.add_argument('--truncate_ep', dest='truncate_ep', default=True,
                                  help='truncate episode run in dataloader in order to do only the necessary steps')
        self.parser.add_argument('--occ_from_depth', dest='occ_from_depth', default=True, action='store_true',
                                help='if enabled, uses only depth to get the ground-projected egocentric grid')

        self.parser.add_argument('--scenes_list', nargs='+')

        self.parser.add_argument('--root_path', type=str, dest='root_path', default="/home/robotlab/")

        # 下面2个是 读取glb的路径
        # 读取json.gz的路径： episodes_path + ep_set + '/'  + cfg.DATASET.SPLIT + "/content/" + self.scene_id + ".json.gz"
        self.parser.add_argument('--episodes_path', type=str, dest='episodes_path', default="dataset/L2M_episodes/")
        self.parser.add_argument('--ep_set', type=str, dest='ep_set', default='objectnav_mp3d_v1', choices=['objectnav_mp3d_v1','v1','v3','v5'])
        # self.parser.add_argument('--episodes_root', type=str, dest='episodes_root', default="")

        #  下面两个是，保存NPZ的路径，同时也是读取glb的路径。
        # 读取glb的路径： root_path + scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb'
        # 保存NPZ的路径： root_path + scenes_dir + episodes_save_dir + split + "/"
        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='$$$$$wtf/dataset/MP3D_dataset/v1/tasks/mp3d_habitat/')
        self.parser.add_argument('--episodes_save_dir', type=str, dest='episodes_save_dir', default="NPZ/")

        # scene_id 是场景的名字

        self.parser.add_argument('--gpu_capacity', type=int, dest='gpu_capacity', default=2)

        self.parser.add_argument('--occupancy_height_thresh', type=float, dest='occupancy_height_thresh', default=-1.0,
                                 help='used when estimating occupancy from depth')

# 每个进程实际执行的任务。
# 每个进程负责一个 scene_id。
def store_episodes(options, config_file, scene_id):
    # 创建保存目录
    episode_save_dir = options.root_path + options.scenes_dir  + options.episodes_save_dir + options.split + "/" + scene_id + "/"
    if not os.path.exists(episode_save_dir):
        os.makedirs(episode_save_dir)

    # 获取已有 episode 列表, 防止重复生成。
    existing_episode_list = os.listdir(episode_save_dir) # keep track of previously saved episodes

    # 初始化数据类. 通过 HabitatDataScene 加载该场景的模拟器；产生所有 episode（轨迹）样本。
    print("     [zhjd-debug] options.episodes_path: ", options.episodes_path)
    print("     [zhjd-debug] options.ep_set: ", options.ep_set)
    options.episodes_root = options.episodes_path + options.ep_set + '/'
    print("     [zhjd-debug] options.episodes_root: ", options.episodes_root)
    data = HabitatDataScene(options, config_file, scene_id=scene_id, existing_episode_list=existing_episode_list)

    print(len(data))

    ep_count = len(existing_episode_list)
    # 遍历生成每个 episode
    for i in range(len(data)):
        # data[i] 返回一个 episode 的所有张量（pose, grid, image 等）。
        ex = data[i]

        if ep_count >= options.max_num_episodes:
            break

        if ex is None:
            continue

        ep_count+=1

        scene_id = ex['scene_id']
        episode_id = ex['episode_id']
        abs_pose = ex['abs_pose']
        ego_grid_crops_spatial = ex['ego_grid_crops_spatial'].cpu()
        step_ego_grid_crops_spatial = ex['step_ego_grid_crops_spatial'].cpu()
        gt_grid_crops_spatial = ex['gt_grid_crops_spatial'].cpu()
        gt_grid_crops_objects = ex['gt_grid_crops_objects'].cpu()

        images = ex['images'].cpu()
        ssegs = ex['ssegs'].cpu()
        depth_imgs = ex['depth_imgs'].cpu()

        # 截取 episode（只保留末尾 N 步）,确保每个 episode 的长度一致（10 步）。
        # options.truncate_ep 是脚本中一个 控制 episode 截断方式 的布尔参数，它直接决定你生成的数据是“取最后几步”还是“取随机一段”。
        if options.truncate_ep: # assumes that the maps were created only up to the desired step
            abs_pose = abs_pose[-options.episode_len:,:]
            ego_grid_crops_spatial = ego_grid_crops_spatial[-options.episode_len:,:,:,:]
            step_ego_grid_crops_spatial = step_ego_grid_crops_spatial[-options.episode_len:,:,:,:]
            gt_grid_crops_spatial = gt_grid_crops_spatial[-options.episode_len:,:,:,:]
            gt_grid_crops_objects = gt_grid_crops_objects[-options.episode_len:,:,:,:]
            images = images[-options.episode_len:,:,:,:]
            ssegs = ssegs[-options.episode_len:,:,:,:]
            depth_imgs = depth_imgs[-options.episode_len:,:,:,:]
        else: # assumes episode was run until its end
            total_episode_len = ego_grid_crops_spatial.shape[0]
            ind = random.randint(0, total_episode_len-options.episode_len-1)
            abs_pose = abs_pose[ind:ind+options.episode_len,:]
            ego_grid_crops_spatial = ego_grid_crops_spatial[ind:ind+options.episode_len,:,:,:]
            step_ego_grid_crops_spatial = step_ego_grid_crops_spatial[ind:ind+options.episode_len,:,:,:]
            gt_grid_crops_spatial = gt_grid_crops_spatial[ind:ind+options.episode_len,:,:,:]
            gt_grid_crops_objects = gt_grid_crops_objects[ind:ind+options.episode_len,:,:,:]
            images = images[ind:ind+options.episode_len,:,:,:]
            ssegs = ssegs[ind:ind+options.episode_len,:,:,:]
            depth_imgs = depth_imgs[ind:ind+options.episode_len,:,:,:]
        
        print('Saving episode', ep_count, 'of id', episode_id, 'scene', scene_id)

        # 保存为 .npz, 每个文件包含：
        # 智能体轨迹 (abs_pose)
        # 空间栅格 (ego_grid_crops_spatial)
        # 目标语义栅格 (gt_grid_crops_objects)
        # RGB、分割、深度图像等
        filepath = episode_save_dir+'ep_'+str(ep_count)+'_'+str(episode_id)+"_"+scene_id
        print('     [zhjd-debug] filepath', filepath)
        np.savez_compressed(filepath+'.npz',
                            abs_pose=abs_pose,
                            ego_grid_crops_spatial=ego_grid_crops_spatial,
                            step_ego_grid_crops_spatial=step_ego_grid_crops_spatial,
                            gt_grid_crops_spatial=gt_grid_crops_spatial,
                            gt_grid_crops_objects=gt_grid_crops_objects,
                            images=images,
                            ssegs=ssegs,
                            depth_imgs=depth_imgs
                            )


if __name__ == '__main__':
    # 启用安全的多进程模式
    mp.set_start_method('forkserver', force=True)
    # 解析命令行参数
    options = Params().parser.parse_args()

    print("options:")
    for k in options.__dict__.keys():
        print(k, options.__dict__[k])


    save_path = options.root_path + options.scenes_dir + options.episodes_save_dir + options.split + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'options.json'), "w") as f:
        json.dump(vars(options), f, indent=4)

    # 根据 split 选择配置文件
    if options.split=="val":
        config_file = "configs/my_objectnav_mp3d_val.yaml"
    elif options.split=="train":
        config_file = "configs/my_objectnav_mp3d_train.yaml"
    else:
        config_file = "configs/my_objectnav_mp3d_test.yaml"

    scene_ids = options.scenes_list

    # Create iterables for map function
    n = len(scene_ids)
    options_list = [options] * n
    config_files = [config_file] * n
    args = [*zip(options_list, config_files, scene_ids)]

    # ✅ 打印每个任务的参数组合
    print("\n===== Scene processing plan =====")
    for idx, (opt, cfg, scene) in enumerate(args):
        print(f"[{idx + 1}] Scene ID: {scene}")
        print(f"    Config file: {cfg}")
        print(f"    Max episodes: {opt.max_num_episodes}")
        print(f"    Episode length: {opt.episode_len}")
        print(f"    Split: {opt.split}")
        print("-" * 60)
    print("=================================\n")

    with Pool(processes=options.gpu_capacity) as pool:
        pool.starmap(store_episodes, args)

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
