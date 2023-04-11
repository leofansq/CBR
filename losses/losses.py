import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as PLT
import numpy as np
import cv2


def _gather_feat(feat, ind, mask=None):
    """Gather feature map.
        Given feature map and index, return indexed feature map.
        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.
        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = (target.lt(1) & target.ge(0)).float()
        ignore_index = target.eq(-1).float() # ignored pixels

        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = torch.log(prediction) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
                        
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        loss = - negative_loss - positive_loss

        return loss, num_positive


class compute_transform_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_transform_losses, self).__init__()
        self.device = device
        self.l1_loss = L1Loss()

    def forward(self, outputs, retransform_output):
        loss = F.l1_loss(outputs, retransform_output, size_average=False)
        return loss

def compute_multibin_losses(vector_ori, gt_ori, num_bin=4):
    gt_ori = gt_ori.view(-1, gt_ori.shape[-1]) # bin1 cls, bin1 offset, bin2 cls, bin2 offst
    vector_ori = vector_ori.view(-1, vector_ori.shape[-1])

    cls_losses = 0
    reg_losses = 0
    reg_cnt = 0
    for i in range(num_bin):
        # bin cls loss
        cls_ce_loss = F.cross_entropy(vector_ori[:, (i * 2) : (i * 2 + 2)], gt_ori[:, i].long(), reduction='none')
        # regression loss
        valid_mask_i = (gt_ori[:, i] == 1)
        cls_losses += cls_ce_loss.mean()
        if valid_mask_i.sum() > 0:
            s = num_bin * 2 + i * 2
            e = s + 2
            pred_offset = F.normalize(vector_ori[valid_mask_i, s : e])
            reg_loss = F.l1_loss(pred_offset[:, 0], torch.sin(gt_ori[valid_mask_i, num_bin + i]), reduction='none') + \
                        F.l1_loss(pred_offset[:, 1], torch.cos(gt_ori[valid_mask_i, num_bin + i]), reduction='none')

            reg_losses += reg_loss.sum()
            reg_cnt += valid_mask_i.sum()

    return cls_losses / num_bin + reg_losses / reg_cnt

class compute_det_losses():
    def __init__(self, device='cuda'):

        self.orien_bin_size = 4
        self.dim_modes = ['exp', False]
        # Reference car size in (length, height, width)
        # for (car, pedestrian, cyclist)
        self.dim_mean = torch.tensor([(3.8840, 1.5261, 1.6286), (0.8423, 1.7607, 0.6602), (1.7635, 1.7372, 0.5968)]).to(device)
        self.dim_std = torch.tensor([(0.4259, 0.1367, 0.1022), (0.2349, 0.1133, 0.1427), (0.1766, 0.0948, 0.1242)]).to(device)

        self.cls_loss = FocalLoss()
        self.reg_loss = F.l1_loss
    
    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
                cls_id: each object id
                dims_offset: dimension offsets, shape = (N, 3)

        Returns:
        '''
        cls_id = cls_id.long()
        cls_dimension_mean = self.dim_mean[cls_id, :].unsqueeze(1)

        if self.dim_modes[0] == 'exp':
            dims_offset = dims_offset.exp()
        if self.dim_modes[1]:
            cls_dimension_std = self.dim_std[cls_id, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
            dimensions = dims_offset * cls_dimension_mean
            
        return dimensions

    def __call__(self, preds, targets):

        t_maps = targets["bev_map"]
        t_boxes = targets["bev_boxes"]
        t_inds = targets["bev_inds"]
        t_masks = targets["bev_masks"]
        t_ori = targets["bev_ori"]
        t_cls = targets["cls_ids"]

        p_maps = preds['det_cls']

        p_loc = preds['det_reg'][:,:3,:,:].permute(0,2,3,1).contiguous()
        p_loc = p_loc.view(p_loc.size(0), -1, p_loc.size(3))
        p_loc = _gather_feat(p_loc, t_inds)

        p_ori = preds['det_reg'][:,6:,:,:].permute(0,2,3,1).contiguous()
        p_ori = p_ori.view(p_ori.size(0), -1, p_ori.size(3))
        p_ori = _gather_feat(p_ori, t_inds)

        p_dim = preds['det_reg'][:,3:6,:,:].permute(0,2,3,1).contiguous()
        p_dim = p_dim.view(p_dim.size(0), -1, p_dim.size(3))
        p_dim = _gather_feat(p_dim, t_inds)
        p_dim = self.decode_dimension(t_cls, p_dim)

        p_reg = torch.cat((p_loc, p_dim), 2)

        map_loss, num_map_pos = self.cls_loss(p_maps, t_maps)
        map_loss = map_loss / torch.clamp(num_map_pos, 1)

        num_reg = t_masks.float().sum()
        t_masks_reg = t_masks.unsqueeze(2).expand_as(t_boxes).float()
        reg_loss = torch.sum(self.reg_loss(p_reg, t_boxes, reduction='none') * t_masks_reg)
        reg_loss = reg_loss / torch.clamp(num_reg, 1)

        t_masks_ori = t_masks.unsqueeze(2).expand_as(p_ori).float()
        ori_loss = compute_multibin_losses(p_ori* t_masks_ori, t_ori, num_bin=self.orien_bin_size)

		# stop when the loss has NaN or Inf
        for v in [map_loss, reg_loss, ori_loss]:
            if torch.isnan(v).sum() > 0:
                pdb.set_trace()
            if torch.isinf(v).sum() > 0:
                pdb.set_trace()

        return map_loss, reg_loss, ori_loss


class compute_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_losses, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss()
        self.det_loss = compute_det_losses(self.device)

    def forward(self, opt, weight, inputs, outputs):
        losses = {}

        losses["bev_seg_loss"] = self.compute_topview_loss(
            outputs["bev_seg"],
            inputs["bev_seg"],
            weight)
        losses["fv_seg_loss"] = self.compute_topview_loss(
            outputs["fv_seg"],
            inputs["fv_seg"],
            weight)

        losses["det_map_loss"], losses["det_reg_loss"], losses["det_ori_loss"] = self.det_loss(outputs, inputs)
        
        losses["loss"] = 1 * losses["bev_seg_loss"] + losses["fv_seg_loss"] +\
                         losses["det_map_loss"] * 1 + losses["det_reg_loss"] * 1 + losses["det_ori_loss"] * 1

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):
        generated_top_view = outputs
        true_top_view = torch.squeeze(true_top_view.long())
        loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., weight]).cuda())
        output = loss(generated_top_view, true_top_view)
        return output.mean()

    def compute_transform_losses(self, outputs, retransform_output):
        loss = self.L1Loss(outputs, retransform_output)
        return loss
