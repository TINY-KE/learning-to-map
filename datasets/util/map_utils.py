import numpy as np
import os
import torch
from models.semantic_grid import SemanticGrid


def get_acc_proj_grid(ego_grid_sseg, pose, abs_pose, crop_size, cell_size):
    grid_dim = (ego_grid_sseg.shape[2], ego_grid_sseg.shape[3])
    # sg.sem_grid will hold the accumulated semantic map at the end of the episode (i.e. 1 map per episode)
    sg = SemanticGrid(1, grid_dim, crop_size[0], cell_size, spatial_labels=ego_grid_sseg.shape[1], object_labels=ego_grid_sseg.shape[1])
    # Transform the ground projected egocentric grids to geocentric using relative pose
    geo_grid_sseg = sg.spatialTransformer(grid=ego_grid_sseg, pose=pose, abs_pose=abs_pose)
    # step_geo_grid contains the map snapshot every time a new observation is added
    step_geo_grid_sseg = sg.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
    # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
    step_ego_grid_sseg = sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=pose, abs_pose=abs_pose)
    return step_ego_grid_sseg

# 把一帧/多帧相机前方的三维点云（由深度转 3D 得到）投影到地面栅格，输出占用/自由/未知的概率图。
# local3D：列表，长度 = 帧数（或时间步数）。每个元素形状约为 N x 3 的张量，列为 (x, y, z)，坐标已对齐到“机器人前方为 +x”的约定（与上游 depth_to_3D 保持一致）。
# grid_dim：地面栅格尺寸 (H, W)。
# cell_size：单个网格边长（米）。
# occupancy_height_thresh：按高度 y阈值分类“占用/自由”的门槛（默认 -0.9m）。
# 返回：ego_grid_occ，形状 K x 3 x H x W，K=len(local3D)；通道含义：
    # 0: unknown/void（未知）
    # 1: occupied（占用）
    # 2: free（自由）
    # 每个像素对三个通道做了归一化，表示概率分布。
def est_occ_from_depth(local3D, grid_dim, cell_size, device, occupancy_height_thresh=-0.9):
    # print("     [zhjd-debug] est_occ_from_depth, len(local3D): ", len(local3D))   经过debug，可知等于1

    # 给每个时间步预分配一张 3 通道 概率栅格。
    ego_grid_occ = torch.zeros((len(local3D), 3, grid_dim[0], grid_dim[1]), dtype=torch.float32, device=device)

    for k in range(len(local3D)):

        local3D_step = local3D[k]

        # 1) 预过滤：只保留“可靠且与地面相关”的点
        # Keep points for which z < 3m (to ensure reliable projection)
        # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
        z = -local3D_step[:,2]
        # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
        y = local3D_step[:,1]
        # 过滤条件：
        # 0.5m < 前向距离 < 3m：去掉太近（容易有噪声/遮挡）的点，也去掉过远（深度不可靠）的点。
        # y < 1m：去掉“天花板/上方”之类的点（y 是高度），仅保留近地面的可用点。
        local3D_step = local3D_step[(z < 3) & (z > 0.5) & (y < 1), :]

        # 2）初始标签与阈值分配（占用/自由/未知）
        # initialize all locations as unknown (void)
        # 先把所有点标为 0 = unknown，再按高度阈值分成：
        occ_lbl = torch.zeros((local3D_step.shape[0], 1), dtype=torch.float32, device=device)

        # threshold height to get occupancy and free labels
        # TODO: occ_lbl = [N, 1]     # 每个点的占用类别: 0=void, 1=occupied, 2=free
        thresh = occupancy_height_thresh
        y = local3D_step[:,1]
        occ_lbl[y>thresh,:] = 1 # 高于阈值 -> 占用(1)
        occ_lbl[y<=thresh,:] = 2    # 低于阈值 -> 自由(2)

        # 3) 投影到 2D 地面栅格（离散化）
        # 把每个 3D 点 (x, z) 落到地面网格 (col, row)（整数索引）。
        # TODO: map_coords = [N, 2] 代表 每个深度点对应的栅格坐标 (col, row)
        map_coords = discretize_coords(x=local3D_step[:,0], z=local3D_step[:,2], grid_dim=grid_dim, cell_size=cell_size)

        # 4）复制式池化”以得到每个像素的类别计数
        ## Replicate label pooling
        # TODO: grid = [3, H, W]     # 三通道栅格：第0层void，第1层occupied，第2层free
        # 用 均匀分布 1/3 初始化三个通道（unknown/occ/free）。这是一个Dirichlet 先验的简化（避免后续全为 0 的除零问题，也让没有观测的格子保持均匀不确定）。
        grid = torch.empty(3, grid_dim[0], grid_dim[1], device=device)
        grid[:] = 1 / 3

        # 如果没有任何点投影到网格，就直接返回“均匀”分布。
        # If the robot does not project any values on the grid, then return the empty grid
        if map_coords.shape[0]==0:
            ego_grid_occ[k,:,:,:] = grid.unsqueeze(0)
            continue

        # 把每个点的落格 (col,row) 与它的类别 cls ∈ {0,1,2} 合并，然后对相同 (col,row,cls) 做 unique 计数。
        # ① 合并 (col, row, cls)
        concatenated = torch.cat([map_coords, occ_lbl.long()], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        # | grid维度        | 代表                              |
        # | ------------- | ------------------------------- |
        # | 第 0 维 (`cls`) | 格子内投影电云的数量           |
        # | 第 1 维 (`row`) | 栅格的 y/z 坐标                      |
        # | 第 2 维 (`col`) | 栅格的 x 坐标                        |
        # TODO: 举例  grid[:, i, j] = [10, 30, 60]   # 该位置有 10 个 void 点，30 个占用点，60 个空闲点
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5

        # 5) 对每个像素做概率归一化
        # | 类别       | 原计数 | 归一化后概率         |
        # | -------- | --- | -------------- |
        # | void     | 10  | 10 / 100 = 0.1 |
        # | occupied | 30  | 30 / 100 = 0.3 |
        # | free     | 60  | 60 / 100 = 0.6 |
        ego_grid_occ[k,:,:,:] = grid / grid.sum(dim=0)

    # ZHJD: 因此ego_grid_occ相同xy处的三个栅格，数值最大的对应的状态是可视化是应该显示的状态
    return ego_grid_occ



def ground_projection(points2D, local3D, sseg, sseg_labels, grid_dim, cell_size):
    ego_grid_sseg = torch.zeros((sseg.shape[0], sseg_labels, grid_dim[0], grid_dim[1]), dtype=torch.float32, device='cuda')

    for i in range(sseg.shape[0]): # sequence length
        sseg_step = sseg[i,:,:,:].unsqueeze(0) # 1 x 1 x H x W
        points2D_step = points2D[i]
        local3D_step = local3D[i]

        # Keep points for which z < 3m (to ensure reliable projection)
        # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
        z = -local3D_step[:,2]
        valid_inds = torch.nonzero(torch.where((z<3) & (z>0.5), 1, 0)).squeeze(dim=1)
        local3D_step = local3D_step[valid_inds,:]
        points2D_step = points2D_step[valid_inds,:]
        # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
        y = local3D_step[:,1]
        valid_inds = torch.nonzero(torch.where(y<1, 1, 0)).squeeze(dim=1)
        local3D_step = local3D_step[valid_inds,:]
        points2D_step = points2D_step[valid_inds,:]

        map_coords = discretize_coords(x=local3D_step[:,0], z=local3D_step[:,2], grid_dim=grid_dim, cell_size=cell_size)

        grid_sseg = label_pooling(sseg_step, points2D_step, map_coords, sseg_labels, grid_dim)
        grid_sseg = grid_sseg.unsqueeze(0)

        ego_grid_sseg[i,:,:,:] = grid_sseg

    return ego_grid_sseg


def label_pooling(sseg, points2D, map_coords, sseg_labels, grid_dim):
    # pool the semantic labels
    # For each bin get the frequencies of the class labels based on the labels projected
    # Each grid location will hold a probability distribution over the semantic labels
    grid = torch.ones((sseg_labels, grid_dim[0], grid_dim[1]), device='cuda')*(1/sseg_labels) # initially uniform distribution over the labels

    # If the robot does not project any values on the grid, then return the empty grid
    if map_coords.shape[0]==0:
        return grid
    pix_x, pix_y = points2D[:,0].long(), points2D[:,1].long()
    pix_lbl = sseg[0, 0, pix_y, pix_x]
    # SPEEDUP if map_coords is sorted, can switch to unique_consecutive
    uniq_rows = torch.unique(map_coords, dim=0)
    for i in range(uniq_rows.shape[0]):
        ucoord = uniq_rows[i,:]
        # indices of where ucoord can be found in map_coords
        ind = torch.nonzero(torch.where((map_coords==ucoord).all(axis=1), 1, 0)).squeeze(dim=1)
        bin_lbls = pix_lbl[ind]
        hist = torch.histc(bin_lbls, bins=sseg_labels, min=0, max=sseg_labels)
        hist = hist + 1e-5 # add a very small number to every location to avoid having 0s
        hist = hist / float(bin_lbls.shape[0])
        grid[:, ucoord[1], ucoord[0]] = hist
    return grid



def discretize_coords(x, z, grid_dim, cell_size, translation=0):
    # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
    # If translation=0, assumes the agent is at the center
    # If we want the agent to be positioned lower then use positive translation. When getting the gt_crop, we need negative translation
    map_coords = torch.zeros((len(x), 2), device='cuda')
    xb = torch.floor(x[:]/cell_size) + (grid_dim[0]-1)/2.0
    zb = torch.floor(z[:]/cell_size) + (grid_dim[1]-1)/2.0 + translation
    xb = xb.int()
    zb = zb.int()
    map_coords[:,0] = xb  # 把所有点的第 0 列（x 轴方向）赋值为 xb（离散化后的 x 索引）；  例如：如果 xb = [2, 4, 6]，则 map_coords[:,0] = [2,4,6]
    map_coords[:,1] = zb  # 把所有点的第 1 列（z 轴方向）赋值为 zb（离散化后的 z 索引）
    # keep bin coords within dimensions 确保坐标不超出地图范围（防止数组越界）。
    map_coords[map_coords>grid_dim[0]-1] = grid_dim[0]-1
    map_coords[map_coords<0] = 0
    return map_coords.long()



def get_gt_crops(abs_pose, pcloud, label_seq_all, agent_height, grid_dim, crop_size, cell_size):
    x_all, y_all, z_all = pcloud[0], pcloud[1], pcloud[2]
    episode_extend = abs_pose.shape[0]
    gt_grid_crops = torch.zeros((episode_extend, 1, crop_size[0], crop_size[1]), dtype=torch.int64)
    for k in range(episode_extend):
        # slice the gt map according to the agent height at every step
        x, y, label_seq = slice_scene(x_all.copy(), y_all.copy(), z_all.copy(), label_seq_all.copy(), agent_height[k])
        gt = get_gt_map(x, y, label_seq, abs_pose=abs_pose[k], grid_dim=grid_dim, cell_size=cell_size)
        _gt_crop = crop_grid(grid=gt.unsqueeze(0), crop_size=crop_size)
        gt_grid_crops[k,:,:,:] = _gt_crop.squeeze(0)
    return gt_grid_crops


def get_gt_map(x, y, label_seq, abs_pose, grid_dim, cell_size):
    # Transform the ground-truth map to align with the agent's pose
    # The agent is at the center looking upwards
    point_map = np.array([x,y])
    rot_mat_abs = np.array([[np.cos(-abs_pose[2]), -np.sin(-abs_pose[2])],[np.sin(-abs_pose[2]),np.cos(-abs_pose[2])]])
    trans_mat_abs = np.array([[-abs_pose[1]],[abs_pose[0]]]) #### This is important, the first index is negative.
    ##rotating and translating point map points
    t_points = point_map - trans_mat_abs
    rot_points = np.matmul(rot_mat_abs,t_points)
    x_abs = torch.tensor(rot_points[0,:], device='cuda')
    y_abs = torch.tensor(rot_points[1,:], device='cuda')

    map_coords = discretize_coords(x=x_abs, z=y_abs, grid_dim=grid_dim, cell_size=cell_size)

    true_seg_grid = torch.zeros((grid_dim[0], grid_dim[1], 1), device='cuda')
    true_seg_grid[map_coords[:,1], map_coords[:,0]] = label_seq

    ### We need to flip the ground truth to align with the observations.
    ### Probably because the -y tp -z is a rotation about x axis which also flips the y coordinate for matteport.
    true_seg_grid = torch.flip(true_seg_grid, dims=[0])
    true_seg_grid = true_seg_grid.permute(2, 0, 1)
    return true_seg_grid


def crop_grid(grid, crop_size):
    # Assume input grid is already transformed such that agent is at the center looking upwards
    grid_dim_h, grid_dim_w = grid.shape[2], grid.shape[3]
    cx, cy = int(grid_dim_w/2.0), int(grid_dim_h/2.0)
    rx, ry = int(crop_size[0]/2.0), int(crop_size[1]/2.0)
    top, bottom, left, right = cx-rx, cx+rx, cy-ry, cy+ry
    return grid[:, :, top:bottom, left:right]

def slice_scene(x, y, z, label_seq, height):
    # z = -z
    # Slice the scene below and above the agent
    below_thresh = height-0.2
    above_thresh = height+2.0
    all_inds = np.arange(y.shape[0])
    below_inds = np.where(z<below_thresh)[0]
    above_inds = np.where(z>above_thresh)[0]
    invalid_inds = np.concatenate( (below_inds, above_inds), 0) # remove the floor and ceiling inds from the local3D points
    inds = np.delete(all_inds, invalid_inds)
    x_fil = x[inds]
    y_fil = y[inds]
    label_seq_fil = torch.tensor(label_seq[inds], dtype=torch.float, device='cuda')
    return x_fil, y_fil, label_seq_fil


def get_explored_grid(grid_sseg, thresh=0.5):
    # Use the ground-projected ego grid to get observed/unobserved grid
    # Single channel binary value indicating cell is observed
    # Input grid_sseg T x C x H x W (can be either H x W or cH x cW)
    # Returns T x 1 x H x W
    T, C, H, W = grid_sseg.shape
    grid_explored = torch.ones((T, 1, H, W), dtype=torch.float32).to(grid_sseg.device)
    grid_prob_max = torch.amax(grid_sseg, dim=1)
    inds = torch.nonzero(torch.where(grid_prob_max<=thresh, 1, 0))
    grid_explored[inds[:,0], 0, inds[:,1], inds[:,2]] = 0
    return grid_explored

