from __future__ import absolute_import, division, print_function

import math
import os
import random
import pdb

import PIL.Image as pil
import matplotlib.pyplot as PLT
import cv2

import numpy as np

import torch
import torch.utils.data as data
from scipy.ndimage.filters import gaussian_filter

from torchvision import transforms

from .utils.kitti_utils import Calibration, read_label, approx_proj_center, refresh_attributes, show_heatmap, show_image_with_boxes, show_edge_heatmap, show_bevmap
from .utils.augmentations import get_composed_augmentations
from .utils.heatmap_coder import gaussian_radius, draw_umich_gaussian

def pil_loader(path):
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')


def process_topview(topview, size):
    topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    topview_n = np.zeros(topview.shape)
    topview_n[topview == 255] = 1  # [1.,0.]
    return topview_n


def resize_topview(topview, size):
    topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    return topview


def process_discr(topview, size):
    topview = resize_topview(topview, size)
    topview_n = np.zeros((size, size, 2))
    topview_n[topview == 255, 1] = 1.
    topview_n[topview == 0, 0] = 1.
    return topview_n


class MonoDataset(data.Dataset):
    def __init__(self, opt, filenames, split='train', is_train=True):
        super(MonoDataset, self).__init__()

        self.opt = opt
        self.data_path = self.opt.data_path
        self.filenames = filenames
        self.split = split
        self.is_train = is_train
        self.height = self.opt.height
        self.width = self.opt.width
        self.interp = pil.ANTIALIAS
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize(
            (self.height, self.width), interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        inputs["color"] = color_aug(self.resize(inputs["color"]))
        for key in inputs.keys():
            if key != "color" and "discr" not in key and key != "filename":
                inputs[key] = process_topview(
                    inputs[key], self.opt.occ_map_size)
            elif key != "filename":
                inputs[key] = self.to_tensor(inputs[key])

    def __len__(self):
        return len(self.filenames)

    def get_color(self, path):
        color = self.loader(path)

        return color

    def get_static(self, path, do_flip):
        tv = self.loader(path)

        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)

        return tv.convert('L')

    def get_dynamic(self, path):
        tv = self.loader(path)

        return tv.convert('L')

    def get_osm(self, path, do_flip):
        osm = self.loader(path)
        return osm

    def get_static_gt(self, path, do_flip):
        tv = self.loader(path)
        return tv.convert('L')

    def get_dynamic_gt(self, path):
        tv = self.loader(path)
        return tv.convert('L')


class KITTIObject(MonoDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIObject, self).__init__(*args, **kwargs)
        self.root_dir = self.opt.data_path
        self.image_dir = os.path.join(self.root_dir, 'image_2')
        self.bev_dir = os.path.join(self.root_dir, 'bev_seg')
        self.fv_dir = os.path.join(self.root_dir, 'fv_seg')
        self.label_dir = os.path.join(self.root_dir, "label_2")
        # self.calib_dir = os.path.join(self.root_dir, "calib")

        self.label_files = [i+".txt" for i in self.filenames]
        self.classes = ['Car', 'Van', 'Bus', 'Truck']

        self.num_classes = 1
        self.max_objs = 100
        self.multibin_size = 4
        self.down_ratio = 4
        self.alpha_centers = np.array([0, np.pi / 2, np.pi, - np.pi / 2])
        self.TYPE_ID_CONVT = {'Car':0}

        self.augmentation = get_composed_augmentations() if self.is_train else None


    def get_image_path(self, frame_index):
        img_path = os.path.join(self.image_dir, "%06d.jpg" % int(frame_index))
        return img_path

    def get_bev_path(self, frame_index):
        seg_path = os.path.join(self.bev_dir, "%06d.png" % int(frame_index))
        return seg_path
    
    def get_fv_path(self, frame_index):
        seg_path = os.path.join(self.fv_dir, "%06d.png" % int(frame_index))
        return seg_path

    def get_static_gt_path(self, root_dir, frame_index):
        pass

    def get_calibration(self, idx, use_right_cam=False):
        calib_filename = os.path.join(self.calib_dir, self.label_files[idx])
        return Calibration(calib_filename, use_right_cam=use_right_cam)

    def get_label_objects(self, idx):
        if self.split != 'test':
            label_filename = os.path.join(self.label_dir, self.label_files[idx])
        return read_label(label_filename)
    
    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        valid_obj_list = []
        for obj in obj_list:
            if obj.type not in type_whitelist:
                continue

            obj.type = 'Car'                
            
            valid_obj_list.append(obj)

        return valid_obj_list
    
    def preprocess(self, inputs, color_aug):
        inputs["color"] = color_aug(self.resize(inputs["color"]))
        for key in inputs.keys():
            if key in ['bev_seg', 'fv_seg']:
                inputs[key] = process_topview(
                    inputs[key], self.opt.occ_map_size)
                inputs[key] = self.to_tensor(inputs[key])
            elif key in ['color']:
                inputs[key] = self.to_tensor(inputs[key])
            elif key not in ['calib', 'filename']:
                    inputs[key] = torch.tensor(inputs[key])
    
    def encode_alpha_multibin(self, alpha, num_bin=2, margin=1 / 6):
		# encode alpha (-PI ~ PI) to 2 classes and 1 regression
        encode_alpha = np.zeros(num_bin * 2)
        bin_size = 2 * np.pi / num_bin # pi
        margin_size = bin_size * margin # pi / 6

        bin_centers = self.alpha_centers
        range_size = bin_size / 2 + margin_size

        offsets = alpha - bin_centers
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi

        for i in range(num_bin):
            offset = offsets[i]
            if abs(offset) < range_size:
                encode_alpha[i] = 1
                encode_alpha[i + num_bin] = offset

        return encode_alpha

    def __getitem__(self, index):
        inputs = {}

        frame_index = self.filenames[index]  # .split()
        # check this part from original code if the dataset is changed
        # folder = self.opt.data_path
        inputs["filename"] = frame_index
        # calib = self.get_calibration(index)

        objs = self.get_label_objects(index) if self.split != 'test' else None
        objs = self.filtrate_objects(objs)

        img = self.get_color(self.get_image_path(frame_index))
        inputs['ori_img'] = np.array(img)

        # random horizontal flip
        if self.augmentation is not None:
            bev_seg = self.get_dynamic(self.get_bev_path(frame_index))
            fv_seg = self.get_dynamic(self.get_fv_path(frame_index))

            # img, bev_seg, fv_seg, objs, calib = self.augmentation(img, bev_seg, fv_seg, objs, calib)
            img, bev_seg, fv_seg, objs = self.augmentation(img, bev_seg, fv_seg, objs)
            inputs["bev_seg"] = bev_seg
            inputs["fv_seg"] = fv_seg
        else:
            bev_seg = self.get_dynamic_gt(self.get_bev_path(frame_index))
            fv_seg = self.get_dynamic_gt(self.get_fv_path(frame_index))
            inputs["bev_seg"] = bev_seg
            inputs["fv_seg"] = fv_seg

        inputs["color"] = img
        inputs["img_shape"] = np.array(img).shape
        # inputs['calib'] = calib

        if self.is_train and random.random() > 0.5:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        ######################### vis ###############################
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        # 3d dimension
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        # 3d location
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        # rotation y
        rotys = np.zeros([self.max_objs], dtype=np.float32)
                
        ######################### bev ###############################
        bev_map = np.zeros([self.num_classes, self.opt.occ_map_size, self.opt.occ_map_size], dtype=np.float32)
        bev_boxes = np.zeros([self.max_objs, 6], dtype=np.float32)
        bev_inds = np.zeros([self.max_objs], dtype=np.int64)
        bev_masks = np.zeros([self.max_objs], dtype=np.float32)
        orientations = np.zeros([self.max_objs, self.multibin_size * 2], dtype=np.float32)

        ######################### cv ################################
        cv_seg = np.zeros([self.num_classes, self.width//self.down_ratio, self.height//self.down_ratio], dtype=np.float32)

        for i, obj in enumerate(objs):
            cls_id = self.TYPE_ID_CONVT[obj.type]
            locs = obj.t.copy()
            locs[1] = locs[1] - obj.h / 2
            x, z, y = locs[0], locs[1], locs[2]
            
            pc_range = np.array([-45.0, 0.0, -5.0, 45.0, 90.0, 5.0])
            if x<pc_range[0] or x>pc_range[3] or y<pc_range[0] or y>pc_range[4]:
                continue

            box2d = obj.box2d.copy()
            box2d /= self.down_ratio
            resize_ratio = [self.width/inputs["img_shape"][0], self.height/inputs["img_shape"][1]]

            cv_seg[cls_id,int(box2d[1]*resize_ratio[0]):int(box2d[3]*resize_ratio[0])+1, int(box2d[0]*resize_ratio[1]):int(box2d[2]*resize_ratio[1])+1] = 1   
            
            # BEV
            
            bev_map_size = np.array([self.opt.occ_map_size, self.opt.occ_map_size])

            bev_r = gaussian_radius(obj.l, obj.w)
            bev_r = max(2, int(bev_r))

            bev_x = np.clip(((x - pc_range[0])*bev_map_size[0]/pc_range[4]), 0, bev_map_size[0]-1)
            bev_y = np.clip(bev_map_size[0]-((y - pc_range[1])*bev_map_size[1]/pc_range[4]), 0, bev_map_size[0]-1)
            bev_c = np.array([bev_x, bev_y])
            bev_c_int = np.array([int(bev_x), int(bev_y)]) 

            bev_map[cls_id] = draw_umich_gaussian(bev_map[cls_id], bev_c_int, bev_r)

            x, y = bev_c_int[0], bev_c_int[1]

            bev_inds[i] = y * bev_map_size[0] + x
            bev_masks[i] = 1
            bev_boxes[i] = np.array([bev_c[0]-x, bev_c[1]-y, z, obj.l, obj.h, obj.w])
            orientations[i] = self.encode_alpha_multibin(obj.alpha, num_bin=self.multibin_size)

            # vis
            reg_mask[i] = 1
            locations[i] = locs
            dimensions[i] = np.array([obj.l, obj.h, obj.w])
            rotys[i] = obj.ry
        # show_heatmap(np.array(img), cv_seg, index=frame_index)
        # show_bevmap(bev_map, index=frame_index)

        inputs["bev_map"] = bev_map
        inputs["bev_inds"] = bev_inds
        inputs["bev_masks"] = bev_masks
        inputs["bev_boxes"] = bev_boxes
        inputs["bev_ori"] = orientations
        inputs["cls_ids"] = 0

        inputs["cv_seg"] = cv_seg

        # vis
        inputs['reg_mask'] = reg_mask
        inputs['locations'] = locations
        inputs['dimensions'] = dimensions
        inputs['rotys'] = rotys
		##############################################################

        self.preprocess(inputs, color_aug)
        return inputs