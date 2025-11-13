
import numpy as np 
import torch
import torch.nn.functional as F


class SemanticGrid(object):
    
    def __init__(self, batch_size, grid_dim, crop_size, cell_size, spatial_labels, object_labels):
        self.grid_dim = grid_dim
        self.cell_size = cell_size
        self.spatial_labels = spatial_labels
        self.object_labels = object_labels
        self.batch_size = batch_size
        self.crop_size = crop_size

        self.crop_start = int( (self.grid_dim[0] / 2) - (self.crop_size / 2) )
        self.crop_end = int( (self.grid_dim[0] / 2) + (self.crop_size / 2) )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # predicted sem grid over entire scene -- initially uniform distribution over the labels
        #  初始化均匀先验, 则sem_grid中的每一个值都为 1/27
        self.sem_grid = torch.ones((self.batch_size, self.object_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        self.sem_grid = self.sem_grid*(1/self.object_labels)

        # observed ground-projected sem grid over entire scene
        # 初始化均匀先验, 则proj_grid中的每一个值都为 0.3333
        self.proj_grid = torch.ones((self.batch_size, self.spatial_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        self.proj_grid = self.proj_grid*(1/self.spatial_labels)
        
        # Maps containing accumulated uncertainty
        self.uncertainty_map = torch.zeros((self.batch_size, 1, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        self.per_class_uncertainty_map = torch.zeros((self.batch_size, self.object_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)

    # 从地面投影网格到地心坐标系的转换
    # Transform each ground-projected grid into geocentric coordinates
    def spatialTransformer(self, grid, pose, abs_pose):
        # Input: 
        # grid -- sequence len x number of classes x grid_dim x grid_dim
        # pose -- sequence len x 3
        # abs_pose -- same as pose

        geo_grid_out = torch.zeros((grid.shape[0], grid.shape[1], self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(grid.device)

        init_pose = abs_pose[0,:] # init absolute pose of each sequence
        init_rot_mat = torch.tensor([[torch.cos(init_pose[2]), -torch.sin(init_pose[2])],
                                        [torch.sin(init_pose[2]),torch.cos(init_pose[2])]], dtype=torch.float32).to(grid.device)

        for j in range(grid.shape[0]): # sequence length

            grid_step = grid[j,:,:,:].unsqueeze(0)
            pose_step = pose[j,:]
        
            rel_coord = torch.tensor([pose_step[1],pose_step[0]], dtype=torch.float32).to(grid.device)
            rel_coord = rel_coord.reshape((2,1))
            rel_coord = torch.matmul(init_rot_mat,rel_coord)
    
            x = 2*(rel_coord[0]/self.cell_size)/(self.grid_dim[0])
            z = 2*(rel_coord[1]/self.cell_size)/(self.grid_dim[1])
    
            angle = pose_step[2]

            trans_theta = torch.tensor( [ [1, -0, x], [0, 1, z] ], dtype=torch.float32 ).unsqueeze(0)
            rot_theta = torch.tensor( [ [torch.cos(angle), -1.0*torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0] ], dtype=torch.float32 ).unsqueeze(0)
            trans_theta = trans_theta.to(grid.device)
            rot_theta = rot_theta.to(grid.device)
            
            trans_disp_grid = F.affine_grid(trans_theta, grid_step.size(), align_corners=False) # get grid translation displacement
            rot_disp_grid = F.affine_grid(rot_theta, grid_step.size(), align_corners=False) # get grid rotation displacement
            
            rot_geo_grid = F.grid_sample(grid_step, rot_disp_grid.float(), align_corners=False ) # apply rotation
            geo_grid = F.grid_sample(rot_geo_grid, trans_disp_grid.float(), align_corners=False) # apply translation

            geo_grid = geo_grid + 1e-12
            geo_grid_out[j,:,:,:] = geo_grid

        return geo_grid_out

    # 这个函数实现了一个空间坐标变换器，将地面投影的网格数据转换到地心坐标系。主要用于人形机器人的路径规划系统，处理在狭窄空间中的坐标变换问题
    # Transform a geocentric map back to egocentric view
    # # 输入参数：
    # grid:     序列长度 × 类别数 × 网格高度 × 网格宽度 的张量
    # pose:     序列长度 × 3 的相对位姿 [x, y, theta]
    # abs_pose: 序列长度 × 3 的绝对位姿 [x, y, theta]
    def rotate_map(self, grid, rel_pose, abs_pose):
            # grid -- sequence len x number of classes x grid_dim x grid_dim
            # rel_pose -- sequence len x 3
            ego_grid_out = torch.zeros((grid.shape[0], grid.shape[1], self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(grid.device)
            init_pose = abs_pose[0,:] # init absolute pose of each sequence
            init_rot_mat = torch.tensor([[torch.cos(init_pose[2]), -torch.sin(init_pose[2])],
                                                    [torch.sin(init_pose[2]),torch.cos(init_pose[2])]], dtype=torch.float32).to(grid.device)
            for i in range(grid.shape[0]): # sequence length
                grid_step = grid[i,:,:,:].unsqueeze(0)
                rel_pose_step = rel_pose[i,:]
                rel_coord = torch.tensor([rel_pose_step[1],rel_pose_step[0]], dtype=torch.float32).to(grid.device)
                rel_coord = rel_coord.reshape((2,1))
                rel_coord = torch.matmul(init_rot_mat,rel_coord)
                x = -2*(rel_coord[0]/self.cell_size)/(self.grid_dim[0])
                z = -2*(rel_coord[1]/self.cell_size)/(self.grid_dim[1])
                angle = -rel_pose_step[2]
                
                trans_theta = torch.tensor( [ [1, -0, x], [0, 1, z] ], dtype=torch.float32 ).unsqueeze(0)
                rot_theta = torch.tensor( [ [torch.cos(angle), -1.0*torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0] ], dtype=torch.float32 ).unsqueeze(0)
                trans_theta = trans_theta.to(grid.device)
                rot_theta = rot_theta.to(grid.device)
                
                trans_disp_grid = F.affine_grid(trans_theta, grid_step.size(), align_corners=False) # get grid translation displacement
                rot_disp_grid = F.affine_grid(rot_theta, grid_step.size(), align_corners=False) # get grid rotation displacement
                trans_ego_grid = F.grid_sample(grid_step, trans_disp_grid.float(), align_corners=False ) # apply translation 
                ego_grid = F.grid_sample(trans_ego_grid, rot_disp_grid.float(), align_corners=False) # apply rotation
                ego_grid_out[i,:,:,:] = ego_grid
            return ego_grid_out
    


    def update_sem_grid_bayes(self, geo_grid):
        # Input geo_grid -- B x T x num_of_classes x grid_dim x grid_dim
        # Update the class probabilities at each location of the grid using Bayes rule
        # geo_grid contains the single view observations of the sequence
        step_geo_grid = torch.zeros((geo_grid.shape[0], geo_grid.shape[1], self.object_labels, 
                                                self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(geo_grid.device)
        for i in range(geo_grid.shape[1]): # sequence length
            new_obsv_grid = geo_grid[:,i,:,:,:]
            # 逐类别乘法更新（贝叶斯乘积）
            mul_probs_grid = new_obsv_grid * self.sem_grid
            # 归一化为概率分布
            normalization_grid = torch.sum(mul_probs_grid, dim=1, keepdim=True)
            self.sem_grid = mul_probs_grid / normalization_grid.repeat(1, self.object_labels, 1, 1)
            # 记录
            step_geo_grid[:,i,:,:,:] = self.sem_grid.clone()
        return step_geo_grid

   
    def update_uncertainty_map_avg(self, geo_grid):
        # Input geo_grid -- B x T x 1 x grid_dim x grid_dim
        # Update the uncertainty estimation at each location
        # Update only the locations where the geo_grid has uncertainty values
        for i in range(geo_grid.shape[1]):
            new_uncertainty_grid = geo_grid[:,i,:,:,:].clone()
            inds = torch.nonzero(new_uncertainty_grid > 1e-7, as_tuple=True)
            current_map = self.uncertainty_map.clone()
            current_map[inds[0],inds[1],inds[2],inds[3]] += new_uncertainty_grid[inds[0],inds[1],inds[2],inds[3]]
            current_map[inds[0],inds[1],inds[2],inds[3]] /= 2
            self.uncertainty_map = current_map


    def update_per_class_uncertainty_map_avg(self, geo_grid):
        # Input geo_grid -- B x T x C x grid_dim x grid_dim
        # Update the per class uncertainty estimation at each location
        # Update only the locations where the geo_grid has uncertainty values
        for i in range(geo_grid.shape[1]):
            new_uncertainty_grid = geo_grid[:,i,:,:,:].clone()
            # 找到当前帧中“有意义的不确定度值”的位置索引。
            # 即，仅对 > 1e-7 的非零元素进行更新，避免更新那些 padding 或未观测区域。
            inds = torch.nonzero(new_uncertainty_grid > 1e-7, as_tuple=True)
            current_map = self.per_class_uncertainty_map.clone()
            current_map[inds[0],inds[1],inds[2],inds[3]] += new_uncertainty_grid[inds[0],inds[1],inds[2],inds[3]]
            current_map[inds[0],inds[1],inds[2],inds[3]] /= 2
            self.per_class_uncertainty_map = current_map

    # 全局融合
    def update_proj_grid_bayes(self, geo_grid):
        # geo_grid 维度含义：[批次大小, 时间步, 类别数, 网格高, 网格宽]   通过print可知， size为[1, 1, 3, 384, 384]
        # Input geo_grid -- B x T (or 1) x num_of_classes x grid_dim x grid_dim
        # Update the ground-projected grid at each location

        # step_geo_grid 用来保存“每个时间步的累计后全局分布快照”（所以有 T 维）。
        step_geo_grid = torch.zeros((geo_grid.shape[0], geo_grid.shape[1], self.spatial_labels, 
                                            self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(geo_grid.device)

        for i in range(geo_grid.shape[1]): # sequence 取第 i 步的新观测（世界系下）。但其实就1步
            new_proj_grid = geo_grid[:,i,:,:,:]
            mul_proj_grid = new_proj_grid * self.proj_grid
            normalization_grid = torch.sum(mul_proj_grid, dim=1, keepdim=True)
            self.proj_grid = mul_proj_grid / normalization_grid.repeat(1, geo_grid.shape[2], 1, 1)
            step_geo_grid[:,i,:,:,:] = self.proj_grid.clone()
        return step_geo_grid


    def register_uncertainty(self, uncertainty_crop, pose, abs_pose):
        # used in the active training
        # assumes batch_size=1
        B, T, _, cH, cW = uncertainty_crop.shape
        ego_uncertainty_map = torch.zeros((T,1,self.grid_dim[0],self.grid_dim[1]), dtype=torch.float32, device=self.device)
        ego_uncertainty_map[:,:, self.crop_start:self.crop_end, self.crop_start:self.crop_end] = uncertainty_crop.squeeze(0)
        geo_uncertainty_maps = self.spatialTransformer(grid=ego_uncertainty_map, pose=pose, abs_pose=abs_pose)
        self.update_uncertainty_map_avg(geo_grid=geo_uncertainty_maps.unsqueeze(0)) # updates sg.uncertainty_map

    # 作用：把每类不确定度的小视野裁剪（crop，机器人坐标系/ego-centric）放进一张全局大小的自车坐标系网格，
    # 再用位姿把它配准到全局/地理坐标系（geo-centric）的语义网格上，最后把这些不确定度按平均规则融合进场景级的不确定度地图里。
    # per_class_uncertainty_crop: 形状 [B, T, C, cH, cW]
    # B 批大小（这里通常就是 1）
    # T 时间步/帧数（例如该 episode 内连续若干步的裁剪）
    # C 语义类别数（例如 27）
    # cH, cW 裁剪区域的高宽（比如 64×64）
    # 函数里紧接着会 squeeze(0)，意味着期望 B=1。
    def register_per_class_uncertainty(self, per_class_uncertainty_crop, pose, abs_pose):
        B, T, C, cH, cW = per_class_uncertainty_crop.shape
        # 在自车坐标系下建立一张大网格（H×W = self.grid_dim），初值全 0（不确定度 0，表示“未知处没有不确定度记录”，而非“确定”——注意后续融合逻辑会处理）。
        ego_per_class_uncertainty_map = torch.zeros((T,C,self.grid_dim[0],self.grid_dim[1]), dtype=torch.float32, device=self.device)
        # 把裁剪块贴回到这张大网格里：
        # 用 crop_start:crop_end 指定裁剪块在大网格中的位置（通常居中，以“机器人在网格中心”为假设）。
        # 赋值后，ego-grid 里只有裁剪区域有数据，其他区域仍为 0。
        ego_per_class_uncertainty_map[:,:, self.crop_start:self.crop_end, self.crop_start:self.crop_end] = per_class_uncertainty_crop.squeeze(0)
        # 把自车坐标系的不确定度图（逐类、逐时刻）转移到全局坐标系
        # pose 是相对位姿序列（t−1→t），abs_pose 是全局位姿（世界坐标系）。模块内会把每个时间步的 ego 网格旋转+平移到全局网格上。
        geo_per_class_uncertainty_maps = self.spatialTransformer(grid=ego_per_class_uncertainty_map, pose=pose, abs_pose=abs_pose)
        # 把配准后的不确定度图送进平均融合器：
        # 这里用的是均值融合（而不是贝叶斯），即：同一网格同一类别来自不同时刻的不确定度，做加权或简单平均，得到稳定的不确定度估计。
        # unsqueeze(0) 补回 B 维度。
        self.update_per_class_uncertainty_map_avg(geo_grid=geo_per_class_uncertainty_maps.unsqueeze(0)) # updates sg.per_class_uncertainty_map


    # 作用：把每类语义概率预测的裁剪块注册到全局语义网格，并通过贝叶斯更新与历史证据融合，得到随时间收敛的语义概率地图。
    def register_sem_pred(self, prediction_crop, pose, abs_pose):
        B, T, C, cH, cW = prediction_crop.shape
        ego_pred_map = torch.ones((T,C,self.grid_dim[0],self.grid_dim[1]), dtype=torch.float32, device=self.device) * (1/C)
        ego_pred_map[:,:, self.crop_start:self.crop_end, self.crop_start:self.crop_end] = prediction_crop.squeeze(0)
        geo_pred_map = self.spatialTransformer(grid=ego_pred_map, pose=pose, abs_pose=abs_pose)
        self.update_sem_grid_bayes(geo_grid=geo_pred_map.unsqueeze(0)) # updates sg.sem_grid
