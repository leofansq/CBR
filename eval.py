import argparse
import os

import CBR

import numpy as np

import torch
from torch.utils.data import DataLoader

import cv2
import tqdm

from utils import mean_IU, mean_precision, BatchCollator
from opt import get_eval_args as get_args

from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map
from CBR.utils.evaluation import generate_kitti_3d_detection, evaluate_python
from CBR.utils.vis_utils import show_image_with_boxes


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def load_model(models, model_path):
    """Load model(s) from disk
    """
    model_path = os.path.expanduser(model_path)

    assert os.path.isdir(model_path), \
        "Cannot find folder {}".format(model_path)
    print("loading model from folder {}".format(model_path))

    for key in models.keys():
        print("Loading {} weights...".format(key))
        path = os.path.join(model_path, "{}.pth".format(key))
        model_dict = models[key].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {
            k: v for k,
                     v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[key].load_state_dict(model_dict)
    return models


def evaluate():
    opt = get_args()
    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir)
        os.makedirs(os.path.join(opt.out_dir, 'det'))

    # Loading Pretarined Model
    models = {}
    models["encoder"] = CBR.Encoder(18, opt.height, opt.width, True)

    models['DecoupleViewProjection'] = CBR.DecoupleViewProjection(in_dim=16)
    models["CrossViewEnhancement"] = CBR.CrossViewEnhancement(64)

    models["bev_decoder"] = CBR.Decoder(models["encoder"].resnet_encoder.num_ch_enc, opt.num_class, "bev_decoder")
    models["fv_decoder"] = CBR.Decoder(models["encoder"].resnet_encoder.num_ch_enc, opt.num_class, "fv_decoder")


    models["det_heads"] = CBR.Bev_predictor(opt.num_class, 64)

    for key in models.keys():
        models[key].to("cuda")

    models = load_model(models, opt.pretrained_path)
    det_infer = CBR.DetInfer('cuda')

    # Loading Validation/Testing Dataset
    # Data Loaders
    dataset = CBR.KITTIObject
    fpath = os.path.join(opt.data_path, "splits", "{}_files.txt")
    test_filenames = readlines(fpath.format("val"))
    test_dataset = dataset(opt, test_filenames, is_train=False)
    collator = BatchCollator()
    test_loader = DataLoader(
        test_dataset,
        1,
        False,
        num_workers=opt.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True)

    bev_iou, bev_mAP = np.array([0., 0.]), np.array([0., 0.])
    fv_iou, fv_mAP = np.array([0., 0.]), np.array([0., 0.])
    for batch_idx, inputs in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)
        
        # Segmentation
        # save_topview(inputs[0]["filename"], outputs["bev_seg"], os.path.join(
        #         opt.out_dir, 'bev_seg', "{}.png".format(inputs[0]["filename"])))
        bev_pred = np.squeeze(torch.argmax(outputs["bev_seg"].detach(), 1).cpu().numpy())
        bev_gt = np.squeeze(inputs[0]["bev_seg"].detach().cpu().numpy())
        bev_iou += mean_IU(bev_pred, bev_gt)
        bev_mAP += mean_precision(bev_pred, bev_gt)
        
        # save_topview(inputs[0]["filename"], outputs["fv_seg"], os.path.join(
        #         opt.out_dir, 'fv_seg', "{}.png".format(inputs[0]["filename"])))
        fv_pred = np.squeeze(torch.argmax(outputs["fv_seg"].detach(), 1).cpu().numpy())
        fv_gt = np.squeeze(inputs[0]["fv_seg"].detach().cpu().numpy())
        fv_iou += mean_IU(fv_pred, fv_gt)
        fv_mAP += mean_precision(fv_pred, fv_gt)

        # Detection
        det_pred = det_infer(outputs, inputs)
        det_pred = det_pred.to(torch.device("cpu"))
        predict_txt = inputs[0]['filename'] + '.txt'
        predict_txt = os.path.join(opt.out_dir, 'det', predict_txt)
        generate_kitti_3d_detection(det_pred, predict_txt)

        # vis_path = os.path.join(opt.out_dir, 'vis')
        # if not os.path.isdir(vis_path):
        #     os.makedirs(vis_path)
        # show_image_with_boxes(det_pred, inputs, vis_path)


    bev_iou /= len(test_loader)
    bev_mAP /= len(test_loader)
    fv_iou /= len(test_loader)
    fv_mAP /= len(test_loader)

    det_results, ret_dict = evaluate_python(label_path=os.path.join(opt.data_path, 'label_2'),
                                        result_path=os.path.join(opt.out_dir, 'det'),
                                        label_split_file=fpath.format("val"),
                                        current_class=0,
                                        metric='R40')
    print ('\n' + det_results)

    output = ("bev: mIOU: %.4f mAP: %.4f | fv: mIOU: %.4f mAP: %.4f " % (bev_iou[1], bev_mAP[1], fv_iou[1], fv_mAP[1]))
    print(output)


def process_batch(opt, models, inputs):
    outputs = {}

    features = models["encoder"](torch.stack([t["color"] for t in inputs]).to('cuda'))

    bev_features, fv_features = models["DecoupleViewProjection"](features)

    outputs["bev_seg"], bev_features = models["bev_decoder"](bev_features)
    outputs["fv_seg"], fv_features = models["fv_decoder"](fv_features)

    bev_features = models["CrossViewEnhancement"](fv_features, bev_features)

    outputs["det_cls"], outputs["det_reg"] = models["det_heads"](bev_features)       
    
    return outputs


def save_topview(idx, tv, name_dest_im):
    tv_np = tv.squeeze().cpu().numpy()
    true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]))
    true_top_view[tv_np[1] > tv_np[0]] = 255
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(name_dest_im, true_top_view)

if __name__ == "__main__":
    evaluate()
