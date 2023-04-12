import argparse
import os

from easydict import EasyDict as edict


def get_args():
    parser = argparse.ArgumentParser(description="Training options")
    parser.add_argument("--dataset", type=str, default="dair_c_inf", help="dair_c_inf or dair_i")
    parser.add_argument("--save_path", type=str, default="./models/", help="Path to save models")
    parser.add_argument("--load_weights_folder", type=str, default="", help="Path to a pretrained model used for initialization")
    parser.add_argument("--model_name", type=str, default="CBR", help="Model Name with specifications")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--global_seed", type=int, default=1234, help="seed")
    parser.add_argument("--batch_size", type=int, default=6, help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--lr_transform", type=float, default=2e-4, help="learning rate")
    parser.add_argument('--lr_steps', default=[40], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument("--scheduler_step_size", type=int, default=5, help="step size for the both schedulers")
    parser.add_argument("--weight", type=float, default=15., help="weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=256, help="size of topview occupancy map")
    parser.add_argument("--num_class", type=int, default=2, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=200, help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=5,help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of cpu workers for dataloaders")
    parser.add_argument('--log_root', type=str, default=os.getcwd() + '/log')
    parser.add_argument('--model_split_save', type=bool, default=True)
    parser.add_argument("--out_dir", type=str, default="output")

    configs = edict(vars(parser.parse_args()))
    configs.data_path = "./datasets/{}/training".format(configs.dataset)

    return configs


def get_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation options")
    parser.add_argument("--dataset", type=str, default="dair_c_inf", help="dair_c_inf or dair_i")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained/dair_c_inf/", help="Path to the pretrained model")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--occ_map_size", type=int, default=256, help="size of topview occupancy map")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of cpu workers for dataloaders")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--num_class", type=int, default=2, help="Number of classes")
    parser.add_argument('--vis', action='store_true', help="visualization")
    configs = edict(vars(parser.parse_args()))
    configs.data_path = "./datasets/{}/training".format(configs.dataset)

    return configs

