import argparse
import os

from PIL import Image, ImageDraw

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="MonoLayout DataPreparation options")
    parser.add_argument("--dataset", type=str, default="dair_c_inf", help="dair_c_inf or dair_i")
    parser.add_argument("--range", type=int, default=90, help="Size of the rectangular grid in metric space (in m)")
    parser.add_argument("--occ_map_size", type=int, default=256, help="Occupancy map size (in pixels)")

    return parser.parse_args()



def get_rect(x, y, width, height, theta):
    rect = np.array([(-width / 2, -height / 2), (width / 2, -height / 2),
                     (width / 2, height / 2), (-width / 2, height / 2),
                     (-width / 2, -height / 2)])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset

    return transformed_rect


def get_3Dbox_to_2Dbox(label_path, length, width, res, out_dir):

    if label_path[-3:] != "txt":
        return

    TopView = np.zeros((int(length / res), int(width / res)))
    labels = open(label_path).read()
    labels = labels.split("\n")
    img = Image.fromarray(TopView)
    for label in labels:
        if label == "":
            continue

        elems = label.split()
        if elems[0] in ['Car', 'Van', 'Bus', 'Truck']:
            center_x = int(float(elems[11]) / res + width / (2 * res))
            center_z = int(float(elems[13]) / res)
            orient = -1 * float(elems[14])

            obj_w = int(float(elems[9]) / res)
            obj_l = int(float(elems[10]) / res)

            rectangle = get_rect(
                center_x, int(
                    length / res) - center_z, obj_l, obj_w, orient)
            draw = ImageDraw.Draw(img)
            draw.polygon([tuple(p) for p in rectangle], fill=255)

    img = img.convert('L')
    file_path = os.path.join(
        out_dir,
        os.path.basename(label_path)[
            :-3] + "png")
    img.save(file_path)
    print("Saved file at %s" % file_path)


if __name__ == "__main__":
    args = get_args()
    args.out_dir = '../datasets/{}/training/bev_seg/'.format(args.dataset)
    args.base_path = "../datasets/{}/training/label_2".format(args.dataset)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    for file_path in os.listdir(args.base_path):
        label_path = os.path.join(args.base_path, file_path)
        get_3Dbox_to_2Dbox(
            label_path,
            args.range,
            args.range,
            args.range/float(args.occ_map_size),
            args.out_dir)