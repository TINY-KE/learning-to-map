
import torch
import torch.nn as nn
import torch.nn.functional as F


class MapPredictorHier(nn.Module):
    def __init__(self, segmentation_model, map_loss_scale, with_img_segm):
        super(MapPredictorHier, self).__init__()
        self._segmentation_model = segmentation_model
        self._map_loss_scale = map_loss_scale
        self.with_img_segm = with_img_segm
        
        self.cel_loss_spatial = nn.CrossEntropyLoss()
        self.cel_loss_objects = nn.CrossEntropyLoss()
        

    def forward(self, batch, is_train=True):

        # 以 agent 为中心裁剪出来的空间语义图 crop。
        step_ego_crops = batch['step_ego_grid_crops_spatial']
        # 维度解释：
        # B：batch size（批次大小）
        # T：时间步数（可能是帧序列）
        # _：通道数（可能是 1 或 3）
        # cH, cW：crop 的高和宽（例如 64x64）
        B, T, _, cH, cW = step_ego_crops.shape # batch, sequence length, _, crop height, crop width

        # 调用子模型进行语义预测
        # 可选的 pred_ego_crops_sseg：图像语义分割（如果启用了 with_img_segm）
        if self.with_img_segm:
            pred_maps_raw_spatial, pred_maps_raw_objects = self._segmentation_model(step_ego_crops, batch['pred_ego_crops_sseg'])  # 这里指的是work1前两个unet组成的网络，而不是语义分割网络. SemMapTester 用的是这个.
        else:
            pred_maps_raw_spatial, pred_maps_raw_objects = self._segmentation_model(step_ego_crops)

        # 获取类别数（C）
        # number of classes for each case
        spatial_C = pred_maps_raw_spatial.shape[1]  # zhjd：按理说是一类，但实际上是三类
        objects_C = pred_maps_raw_objects.shape[1]  # zhjd：27类物体

        # TODO: 原始 logits 和 softmax  是什么意思？
        # logits	模型输出的“原始分数”，还没归一化成概率.               用途：用于 loss 计算更稳定            例如：logits = [2.1, 0.3, -1.2]
        # softmax	把 logits 转换成“概率分布”，每个类别的概率总和为 1     用途：用 argmax 选择最大概率的类别    例如：probs = softmax(logits) = [0.81, 0.15, 0.04]

        # reshape logits，使其具有 [B, T, C, H, W] 的标准格式
        # Get a prob distribution over the labels
        pred_maps_raw_spatial = pred_maps_raw_spatial.view(B,T,spatial_C,cH,cW)  # .view(...) 是 PyTorch 中用来“重新改变张量形状”的函数，相当于 NumPy 中的 .reshape()。
        pred_maps_raw_objects = pred_maps_raw_objects.view(B,T,objects_C,cH,cW)

        # TODO: 应用 Softmax → 得到概率分布图。对类别维度（dim=2）做 softmax，将 logits 转换为概率。
        # 这一步非常关键，因为后续会用这些概率来做分类（argmax）或评估（IoU、F1等）。
        pred_maps_spatial = F.softmax(pred_maps_raw_spatial, dim=2)
        pred_maps_objects = F.softmax(pred_maps_raw_objects, dim=2)

        output = {'pred_maps_raw_spatial':pred_maps_raw_spatial,
                  'pred_maps_raw_objects':pred_maps_raw_objects,
                  'pred_maps_spatial':pred_maps_spatial,
                  'pred_maps_objects':pred_maps_objects}
        return output

    
    def loss_cel(self, batch, pred_outputs):
        pred_maps_raw_spatial = pred_outputs['pred_maps_raw_spatial']
        pred_maps_raw_objects = pred_outputs['pred_maps_raw_objects']
        B, T, spatial_C, cH, cW = pred_maps_raw_spatial.shape
        objects_C = pred_maps_raw_objects.shape[2]

        gt_crops_spatial, gt_crops_objects = batch['gt_grid_crops_spatial'], batch['gt_grid_crops_objects']
        pred_map_loss_spatial = self.cel_loss_spatial(input=pred_maps_raw_spatial.view(B*T,spatial_C,cH,cW), target=gt_crops_spatial.view(B*T,cH,cW))
        pred_map_loss_objects = self.cel_loss_objects(input=pred_maps_raw_objects.view(B*T,objects_C,cH,cW), target=gt_crops_objects.view(B*T,cH,cW))
        
        pred_map_err_spatial = pred_map_loss_spatial.clone().detach()
        pred_map_err_objects = pred_map_loss_objects.clone().detach()

        output={}
        output['pred_map_err_spatial'] = pred_map_err_spatial
        output['pred_map_err_objects'] = pred_map_err_objects
        output['pred_map_loss_spatial'] = self._map_loss_scale * pred_map_loss_spatial
        output['pred_map_loss_objects'] = self._map_loss_scale * pred_map_loss_objects
        return output