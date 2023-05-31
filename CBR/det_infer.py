from re import L
import torch
import pdb
import math
import numpy as np

from torch import nn
from .utils.kitti_utils import show_bevmap
from .utils.model_utils import nms_hm


class DetInfer(nn.Module):
	def __init__(self, device='cuda'):
		super(DetInfer, self).__init__()
		self.max_detection = 100
		self.score_threshold = 0.1 #cfg.TEST.DETECTIONS_THRESHOLD
		self.pc_range = [-45.0, 0.0, -5.0, 45.0, 90.0, 5.0]
		self.bev_map_size = [256, 256]
		self.orien_bin_size = 4
		self.alpha_centers = torch.tensor([0, np.pi / 2, np.pi, - np.pi / 2]).to(device)
		self.dim_modes = ['exp', False]
		self.dim_mean = torch.tensor([(3.8840, 1.5261, 1.6286), (0.8423, 1.7607, 0.6602), (1.7635, 1.7372, 0.5968)]).to(device)
		self.dim_std = torch.tensor([(0.4259, 0.1367, 0.1022), (0.2349, 0.1133, 0.1427), (0.1766, 0.0948, 0.1242)]).to(device)
	

	def _gather_feat(self, feats, inds, feat_masks=None):
		"""Given feats and indexes, returns the gathered feats.
		Args:
			feats (torch.Tensor): Features to be transposed and gathered
				with the shape of [B, 2, W, H].
			inds (torch.Tensor): Indexes with the shape of [B, N].
			feat_masks (torch.Tensor, optional): Mask of the feats.
				Default: None.
		Returns:
			torch.Tensor: Gathered feats.
		"""
		dim = feats.size(2)
		inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
		feats = feats.gather(1, inds)
		if feat_masks is not None:
			feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
			feats = feats[feat_masks]
			feats = feats.view(-1, dim)
		return feats


	def _topk(self, scores, K=80):
		"""Get indexes based on scores.
		Args:
			scores (torch.Tensor): scores with the shape of [B, N, W, H].
			K (int, optional): Number to be kept. Defaults to 80.
		Returns:
			tuple[torch.Tensor]
				torch.Tensor: Selected scores with the shape of [B, K].
				torch.Tensor: Selected indexes with the shape of [B, K].
				torch.Tensor: Selected classes with the shape of [B, K].
				torch.Tensor: Selected y coord with the shape of [B, K].
				torch.Tensor: Selected x coord with the shape of [B, K].
		"""
		batch, cat, height, width = scores.size()

		topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

		topk_inds = topk_inds % (height * width)
		topk_ys = (topk_inds.float() /
					torch.tensor(width, dtype=torch.float)).int().float()
		topk_xs = (topk_inds % width).int().float()

		topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
		topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
		topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
										topk_ind).view(batch, K)
		topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
									topk_ind).view(batch, K)
		topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
									topk_ind).view(batch, K)

		return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
	
	def _transpose_and_gather_feat(self, feat, ind):
		"""Given feats and indexes, returns the transposed and gathered feats.
		Args:
			feat (torch.Tensor): Features to be transposed and gathered
				with the shape of [B, 2, W, H].
			ind (torch.Tensor): Indexes with the shape of [B, N].
		Returns:
			torch.Tensor: Transposed and gathered feats.
		"""
		feat = feat.permute(0, 2, 3, 1).contiguous()
		feat = feat.view(feat.size(0), -1, feat.size(3))
		feat = self._gather_feat(feat, ind)
		return feat
	
	def convertRot2Alpha(self, ry3d, z3d, x3d):
		alpha = ry3d - np.arctan(x3d/z3d)
		
		# while alpha > np.pi: alpha -= np.pi * 2
		# while alpha < (-np.pi): alpha += np.pi * 2

		return alpha
	
	def roty(self, t):
		""" Rotation about the y-axis. """
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
	
	def project_8p_to_4p(self, pts_2d):
		x0 = np.min(pts_2d[:, 0])
		x1 = np.max(pts_2d[:, 0])
		y0 = np.min(pts_2d[:, 1])
		y1 = np.max(pts_2d[:, 1])
		x0 = max(np.zeros_like(x0), x0)
		# x1 = min(x1, proj.image_width)
		y0 = max(np.zeros_like(y0), y0)
		# y1 = min(y1, proj.image_height)
		return np.array([x0, y0, x1, y1])
	
	def compute_box_3d(self, ry, dim, x, y, z, calib):
		""" Takes an object and a projection matrix (P) and projects the 3d
			bounding box into the image plane.
			Returns:
				corners_2d: (8,2) array in left image coord.
				corners_3d: (8,3) array in in rect camera coord.
		"""
		boxes_2d = []
		for i in range(len(ry)):
			# compute rotational matrix around yaw axis
			R = self.roty(ry[i])

			# 3d bounding box dimensions h w l
			l = dim[i][2]
			w = dim[i][1]
			h = dim[i][0]

			# 3d bounding box corners
			x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
			y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
			z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
			# rotate and translate 3d bounding box
			corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
			corners_3d[0, :] = corners_3d[0, :] + x[i]
			corners_3d[1, :] = corners_3d[1, :] + y[i]
			corners_3d[2, :] = corners_3d[2, :] + z[i]

			# project the 3d bounding box into the image plane
			corners_2d, _ = calib.project_velo_to_image(np.transpose(corners_3d))
			try:
				corners_2d = self.project_8p_to_4p(corners_2d)
			except:
				pdb.set_trace()
			# print 'corners_2d: ', corners_2d
			boxes_2d.append(corners_2d)
		boxes_2d = np.vstack(boxes_2d)
		return boxes_2d
	
	def decode_axes_orientation(self, vector_ori, loc_x, loc_z):
		'''
		retrieve object orientation
		Args:
				vector_ori: local orientation in [axis_cls, head_cls, sin, cos] format
				locations: object location

		'''
		
		pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
		pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
		orientations = vector_ori.new_zeros(vector_ori.shape[0])
		for i in range(self.orien_bin_size):
			mask_i = (pred_bin_cls.argmax(dim=1) == i)
			s = self.orien_bin_size * 2 + i * 2
			e = s + 2
			pred_bin_offset = vector_ori[mask_i, s : e]
			orientations[mask_i] = torch.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]

		rays = torch.atan2(loc_x, loc_z).squeeze(-1)
		alphas = orientations
		rotys = alphas + rays

		larger_idx = (rotys > np.pi).nonzero()
		small_idx = (rotys < -np.pi).nonzero()
		if len(larger_idx) != 0:
				rotys[larger_idx] -= 2 * np.pi
		if len(small_idx) != 0:
				rotys[small_idx] += 2 * np.pi

		larger_idx = (alphas > np.pi).nonzero()
		small_idx = (alphas < -np.pi).nonzero()
		if len(larger_idx) != 0:
				alphas[larger_idx] -= 2 * np.pi
		if len(small_idx) != 0:
				alphas[small_idx] += 2 * np.pi
		return rotys, alphas
	
	def decode_dimension(self, cls_id, dims_offset):
		'''
		retrieve object dimensions
		Args:
				cls_id: each object id
				dims_offset: dimension offsets, shape = (N, 3)

		Returns:

		'''
		cls_id = cls_id.flatten().long()
		cls_dimension_mean = self.dim_mean[cls_id, :]

		if self.dim_modes[0] == 'exp':
			dims_offset = dims_offset.exp()

		if self.dim_modes[1]:
			cls_dimension_std = self.dim_std[cls_id, :]
			dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
		else:
			dimensions = dims_offset * cls_dimension_mean
			
		return dimensions

	def forward(self, predictions, targets):
		pred_map, pred_reg = nms_hm(predictions['det_cls']), predictions['det_reg']
		# calib = targets[0]["calib"]

		# # debug
		# bev_map = nms_hm((targets[0].get_field("bev_map").unsqueeze(0)))
		# scores, inds, clses, ys, xs = self._topk(bev_map, K=self.max_detection)

		# show_bevmap(pred_map[0].cpu().numpy(), index=targets[0]["filename"])

		batch, _, _, _ = pred_map.size()
		scores, inds, clses, ys, xs = self._topk(pred_map, K=self.max_detection)

		#[bev_c[0]-x, bev_c[0]-y, z, obj.l, obj.h, obj.w, np.sin(rot_y), np.cos(rot_y)]
		reg = pred_reg[:, :2]
		z = pred_reg[:, 2].unsqueeze(1)
		dim = pred_reg[:, 3:6]
		ori = pred_reg[:, 6:]


		reg = self._transpose_and_gather_feat(reg, inds)
		reg = reg.view(batch, self.max_detection, 2)
		xs = xs.view(batch, self.max_detection, 1) + reg[:, :, 0:1]
		ys = ys.view(batch, self.max_detection, 1) + reg[:, :, 1:2]

		ori = self._transpose_and_gather_feat(ori, inds)
		ori = ori.view(batch, self.max_detection, 16)

		# dim of the box
		dim = self._transpose_and_gather_feat(dim, inds)
		dim = dim.view(batch, self.max_detection, 3)
		dim = self.decode_dimension(clses, dim)

		# height in the bev
		z = self._transpose_and_gather_feat(z, inds)
		z = z.view(batch, self.max_detection, 1)
		z += dim[:,:, 1].unsqueeze(-1) / 2
		dim = dim.roll(shifts=-1, dims=2)

		# class label
		clses = clses.view(batch, self.max_detection).float()
		scores = scores.view(batch, self.max_detection)

		xs = xs.view(
			batch, self.max_detection,
			1) * (self.pc_range[3]-self.pc_range[0]) / self.bev_map_size[0] + self.pc_range[0]
		ys = (self.bev_map_size[1]-ys.view(
			batch, self.max_detection,
			1)) * (self.pc_range[4]-self.pc_range[1]) / self.bev_map_size[1] + self.pc_range[1]
		
		final_box_preds = torch.cat([dim, xs, z, ys], dim=2)

		final_scores = scores
		final_preds = clses

		# use score threshold
		if self.score_threshold is not None:
			thresh_mask = final_scores > self.score_threshold

		results = None
		for i in range(batch):
			boxes = final_box_preds[i, thresh_mask[i]]
			ori = ori[i, thresh_mask[i]]
			scores = final_scores[i, thresh_mask[i]].unsqueeze(-1)
			labels = final_preds[i, thresh_mask[i]].unsqueeze(-1)

			x = boxes[:, -3].unsqueeze(-1)
			y = boxes[:, -2].unsqueeze(-1)
			z = boxes[:, -1].unsqueeze(-1)
			dim = boxes[:, :3]
			rot, alpha = self.decode_axes_orientation(ori, x, z)
			# if x.shape[0] == 0:
			# 	boxes2d = torch.zeros([0,4]).cuda()
			# else:
			# 	boxes2d = torch.from_numpy(self.compute_box_3d(rot.cpu().numpy(), dim.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), calib)).cuda()
			boxes2d = torch.ones([boxes.shape[0], 4]).cuda()
			boxes2d[:,3] = boxes2d[:,3]*100
			result = torch.cat([labels, alpha.unsqueeze(-1), boxes2d, boxes, rot.unsqueeze(-1), scores], dim=1)

			if results:
				results = torch.cat([results, result], dim=0)
			else:
				results = result

		return results
