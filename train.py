import argparse
import os
import random
import time
from pathlib import Path

import CBR

import numpy as np
import copy
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import autograd
from torch.optim.lr_scheduler import ExponentialLR
# from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
# from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map

from opt import get_args

import tqdm

from losses import compute_losses
from utils import mean_IU, mean_precision, BatchCollator
from CBR.utils.evaluation import generate_kitti_3d_detection, evaluate_python
from CBR.utils.vis_utils import show_image_with_boxes


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class Trainer:
    def __init__(self):
        self.opt = get_args()
        if not os.path.isdir(self.opt.out_dir):
            os.makedirs(self.opt.out_dir)
            os.makedirs(os.path.join(self.opt.out_dir, 'det'))
        self.device = "cuda"
        self.seed = self.opt.global_seed
        if self.seed != 0:
            self.set_seed()  # set seed

        self.models = {}
        self.inputs = {}
        self.parameters_to_train = []
        self.transform_parameters_to_train = []
        self.detection_parameters_to_train = []
        self.base_parameters_to_train = []
        self.parameters_to_train = []
        self.parameters_to_train_D = []
        self.weight = self.opt.weight

        self.criterion_d = nn.BCEWithLogitsLoss()
        self.criterion = compute_losses(self.device)

        self.create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.epoch = 0
        self.start_epoch = 0
        self.scheduler = 0

        # Save log and models path
        self.opt.log_root = self.opt.log_root
        self.opt.save_path = self.opt.save_path
        self.writer = SummaryWriter(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time))
        self.log = open(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time, '%s.csv' % self.opt.model_name), 'w')
        self.det_output = os.path.join(self.opt.out_dir, 'det')

        # Initializing models
        self.models["encoder"] = CBR.Encoder(18, self.opt.height, self.opt.width, True)

        self.models['DecoupleViewProjection'] = CBR.DecoupleViewProjection(in_dim=16)
        self.models["CrossViewEnhancement"] = CBR.CrossViewEnhancement(64)

        self.models["bev_decoder"] = CBR.Decoder(self.models["encoder"].resnet_encoder.num_ch_enc, self.opt.num_class, "bev_decoder")
        self.models["fv_decoder"] = CBR.Decoder(self.models["encoder"].resnet_encoder.num_ch_enc, self.opt.num_class, "fv_decoder")


        self.models["det_heads"] = CBR.Bev_predictor(self.opt.num_class, 64)

        self.det_infer = CBR.DetInfer(self.device)

        for key in self.models.keys():
            self.models[key].to(self.device)
            if "transform" in key:
                self.transform_parameters_to_train += list(self.models[key].parameters())
            else:
                self.base_parameters_to_train += list(self.models[key].parameters())
        self.parameters_to_train = [
            {"params": self.transform_parameters_to_train, "lr": self.opt.lr_transform},
            {"params": self.base_parameters_to_train, "lr": self.opt.lr},]

        # Optimization
        self.model_optimizer = optim.Adam(self.parameters_to_train)
        # self.scheduler = ExponentialLR(self.model_optimizer, gamma=0.99)
        # self.scheduler = StepLR(self.model_optimizer, step_size=step_size, gamma=0.65)
        # self.scheduler = MultiStepLR(self.model_optimizer, milestones=self.opt.lr_steps, gamma=0.1)
        # self.scheduler = CosineAnnealingLR(self.model_optimizer, T_max=15)  # iou 35.55

        # Data Loaders
        self.dataset = CBR.KITTIObject
        self.fpath = os.path.join(self.opt.data_path, "splits", "{}_files.txt")

        train_filenames = readlines(self.fpath.format("train"))
        val_filenames = readlines(self.fpath.format("val"))
        self.val_filenames = val_filenames
        self.train_filenames = train_filenames

        train_dataset = self.dataset(self.opt, train_filenames)
        val_dataset = self.dataset(self.opt, val_filenames, is_train=False)

        collator = BatchCollator()
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True)
        self.val_loader = DataLoader(
            val_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True)
        
        # Load weights
        if self.opt.load_weights_folder != "":
            self.load_model()

        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))

    def train(self):
        if not os.path.isdir(self.opt.log_root):
            os.mkdir(self.opt.log_root)

        for self.epoch in range(self.start_epoch, self.opt.num_epochs + 1):
            self.adjust_learning_rate(self.model_optimizer, self.epoch, self.opt.lr_steps)
            loss = self.run_epoch()
            output = ("Epoch: %d | lr:%.7f | Loss: %.4f | bev seg Loss: %.4f | fv seg Loss: %.4f || det map Loss: %.4f | det reg Loss: %.4f | det ori Loss: %.4f"
                      % (self.epoch, self.model_optimizer.param_groups[-1]['lr'], loss["loss"], loss["bev_seg_loss"], loss["fv_seg_loss"], loss["det_map_loss"], loss["det_reg_loss"], loss["det_ori_loss"]))
            
            print(output)
            self.log.write(output + '\n')
            self.log.flush()
            for loss_name in loss:
                self.writer.add_scalar(loss_name, loss[loss_name], global_step=self.epoch)
            if self.epoch % self.opt.log_frequency == 0:
                self.validation(self.log)
                if self.opt.model_split_save:
                    self.save_model()
        self.save_model()

    def process_batch(self, inputs, validation=False):
        outputs = {}

        self.inputs['color'] = torch.stack([t["color"] for t in inputs]).to(self.device)
        self.inputs['bev_seg'] = torch.stack([t["bev_seg"] for t in inputs]).to(self.device)
        self.inputs['fv_seg'] = torch.stack([t["fv_seg"] for t in inputs]).to(self.device)
        self.inputs['filename'] = [t["filename"] for t in inputs]
        self.inputs['bev_map'] = torch.stack([t["bev_map"] for t in inputs]).to(self.device)
        self.inputs['bev_inds'] = torch.stack([t["bev_inds"] for t in inputs]).to(self.device)
        self.inputs['bev_masks'] = torch.stack([t["bev_masks"] for t in inputs]).to(self.device)
        self.inputs['bev_boxes'] = torch.stack([t["bev_boxes"] for t in inputs]).to(self.device)
        self.inputs['bev_ori'] = torch.stack([t["bev_ori"] for t in inputs]).to(self.device)
        self.inputs['cls_ids'] = torch.stack([t["cls_ids"] for t in inputs]).to(self.device)
        self.inputs['cv_seg'] = torch.stack([t["cv_seg"] for t in inputs]).to(self.device)

        features = self.models["encoder"](self.inputs["color"])

        bev_features, fv_features = self.models["DecoupleViewProjection"](features)

        if validation:
            outputs["bev_seg"], bev_features = self.models["bev_decoder"](bev_features, False)
            outputs["fv_seg"], fv_features = self.models["fv_decoder"](fv_features, False)
            bev_features = self.models["CrossViewEnhancement"](fv_features, bev_features)
        else:
            outputs["bev_seg"], bev_features = self.models["bev_decoder"](bev_features)
            outputs["fv_seg"], fv_features = self.models["fv_decoder"](fv_features)
            bev_features = self.models["CrossViewEnhancement"](fv_features, bev_features)

        outputs["det_cls"], outputs["det_reg"] = self.models["det_heads"](bev_features)       

        if validation:
            return outputs
        losses = self.criterion(self.opt, self.weight, self.inputs, outputs)

        return outputs, losses

    def run_epoch(self):
        self.model_optimizer.step()
        loss = {
            "loss": 0.0,
            "bev_seg_loss": 0.0,
            "fv_seg_loss": 0.0,
            "det_map_loss": 0.0,
            "det_reg_loss": 0.0,
            "det_ori_loss": 0.0,
        }
        accumulation_steps = 1
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()

            losses["loss"] = losses["loss"] / accumulation_steps
            losses["loss"].backward()
            self.model_optimizer.step()

            for loss_name in losses:
                loss[loss_name] += losses[loss_name].item()
        # self.scheduler.step()
        for loss_name in loss:
            loss[loss_name] /= len(self.train_loader)
        return loss

    def validation(self, log):
        bev_iou, bev_mAP = np.array([0., 0.]), np.array([0., 0.])
        fv_iou, fv_mAP = np.array([0., 0.]), np.array([0., 0.])
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.val_loader)):
            with torch.no_grad():
                outputs = self.process_batch(inputs, True)
            
            # Segmentation
            bev_pred = np.squeeze(torch.argmax(outputs["bev_seg"].detach(), 1).cpu().numpy())
            bev_gt = np.squeeze(self.inputs["bev_seg"].detach().cpu().numpy())
            bev_iou += mean_IU(bev_pred, bev_gt)
            bev_mAP += mean_precision(bev_pred, bev_gt)
            
            fv_pred = np.squeeze(torch.argmax(outputs["fv_seg"].detach(), 1).cpu().numpy())
            fv_gt = np.squeeze(self.inputs["fv_seg"].detach().cpu().numpy())
            fv_iou += mean_IU(fv_pred, fv_gt)
            fv_mAP += mean_precision(fv_pred, fv_gt)

            # Detection
            det_pred = self.det_infer(outputs, inputs)
            det_pred = det_pred.to(torch.device("cpu"))
            predict_txt = inputs[0]['filename'] + '.txt'
            predict_txt = os.path.join(self.det_output, predict_txt)
            generate_kitti_3d_detection(det_pred, predict_txt)

            # show_image_with_boxes(det_pred, inputs)

        bev_iou /= len(self.val_loader)
        bev_mAP /= len(self.val_loader)
        fv_iou /= len(self.val_loader)
        fv_mAP /= len(self.val_loader)

        det_results, ret_dict = evaluate_python(label_path=os.path.join(self.opt.data_path, 'label_2'),
                                        result_path=os.path.join(self.opt.out_dir, 'det'),
                                        label_split_file=self.fpath.format("val"),
                                        current_class=0,
                                        metric='R40')
        print ('\n' + det_results)

        output = ("Epoch: %d | bev: mIOU: %.4f mAP: %.4f | fv: mIOU: %.4f mAP: %.4f " % (self.epoch, bev_iou[1], bev_mAP[1], fv_iou[1], fv_mAP[1]))
        print(output)
        log.write(output + '\n')
        log.write(det_results + '\n')
        log.flush()

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            self.opt.model_name,
            "weights_{}".format(
                self.epoch)
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            state_dict['epoch'] = self.epoch
            if model_name == "encoder":
                state_dict["height"] = self.opt.height
                state_dict["width"] = self.opt.width

            torch.save(state_dict, model_path)
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_path)

    def load_model(self):
        """
        Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for key in self.models.keys():
            print("Loading {} weights...".format(key))
            path = os.path.join(self.opt.load_weights_folder,"{}.pth".format(key))
            model_dict = self.models[key].state_dict()
            pretrained_dict = torch.load(path)
            if 'epoch' in pretrained_dict:
                self.start_epoch = pretrained_dict['epoch']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[key].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def adjust_learning_rate(self, optimizer, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        decay = round(decay, 2)
        lr = self.opt.lr * decay
        lr_transform = self.opt.lr_transform * decay
        decay = self.opt.weight_decay
        optimizer.param_groups[0]['lr'] = lr_transform
        optimizer.param_groups[1]['lr'] = lr
        optimizer.param_groups[0]['weight_decay'] = decay
        optimizer.param_groups[1]['weight_decay'] = decay

    def set_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    start_time = time.ctime()
    print(start_time)
    trainer = Trainer()
    trainer.train()
    end_time = time.ctime()
    print(end_time)
