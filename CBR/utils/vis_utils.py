import numpy as np
import pdb
import cv2

# from .visualization.visualizer import Visualizer
from .kitti_utils import draw_projected_box3d, init_bev_image, draw_bev_box3d


def box3d_to_corners(locs, dims, roty):
	# 3d bbox template
	h, w, l = dims
	x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
	y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
	z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

	# rotation matirx
	R = np.array([[np.cos(roty), 0, np.sin(roty)],
				  [0, 1, 0],
				  [-np.sin(roty), 0, np.cos(roty)]])

	corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
	corners3d = np.dot(R, corners3d).T
	corners3d = corners3d + locs

	return corners3d

def show_image_with_boxes(output, target, vis_path):
    # output Tensor:
    # clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
    image = target[0]['ori_img'].numpy().astype(np.uint8)
    output = output.cpu().float().numpy()

    # filter results with visualization threshold
    vis_thresh = 0.1
    output = output[output[:, -1] > vis_thresh]
    ID_TYPE_CONVERSION = {0:'Car'}

	# predictions
    clses = output[:, 0]
    box2d = output[:, 2:6]
    dims = output[:, 6:9]
    locs = output[:, 9:12]
    rotys = output[:, 12]
    score = output[:, 13]

    # ground-truth
    valid_mask = target[0]['reg_mask']
    num_gt = valid_mask.sum()
    gt_locs = target[0]['locations'][valid_mask]
    gt_dims = target[0]['dimensions'][valid_mask]
    gt_rotys = target[0]['rotys'][valid_mask]

    # print('detections / gt objs: {} / {}'.format(box2d.shape[0], num_gt))

    img3 = image.copy() # for 3d bbox
    img4 = init_bev_image() # for bev

    font = cv2.FONT_HERSHEY_SIMPLEX
    pred_color = (0, 255, 0)
    gt_color = (255, 0, 0)

    # plot prediction 
    for i in range(box2d.shape[0]):

        corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
        img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=pred_color, scores=None)

    # plot ground-truth
    for i in range(num_gt):

        # 3d bbox template
        h, l, w = gt_dims[i]

        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotation matirx
        roty = gt_rotys[i]
        R = np.array([[np.cos(roty), 0, np.sin(roty)],
                        [0, 1, 0],
                        [-np.sin(roty), 0, np.cos(roty)]])

        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + gt_locs[i].numpy() + np.array([0, h / 2, 0]).reshape(1, 3)

        img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=gt_color, scores=None)

    img4 = cv2.resize(img4, (img3.shape[0], img3.shape[0]))
    stack_img = np.concatenate([img3, img4], axis=1)

    cv2.imwrite("{}/{}.png".format(vis_path, target[0]['filename']), cv2.cvtColor(stack_img,cv2.COLOR_RGB2BGR))