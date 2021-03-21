import os
from collections import OrderedDict
import argparse
import datetime
import numpy as np
from sys import exit
from time import time
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.dss import DSS_Net
from models.UNet import UNet
from losses import FocalLoss, DiceCoeff, iou_pytorch
from augmentations import SyntheticAugmentation
from SyntheticSmokeDataset import SyntheticSmokeTrain, SmokeDataset, SimpleSmokeTrain, SimpleSmokeVal

import pynvml
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)


parser = argparse.ArgumentParser(description="ALERTWildfire Smoke Segmentation",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--data_root', help='path to test imgs', required=True)
parser.add_argument('--num_test_images', type=int, help='number of images to test', required=True)
parser.add_argument('--checkpoint_path', help='path to model checkpoint', required=True)
parser.add_argument('--model_name', default='unet')
parser.add_argument('--cuda', type=bool, default=True, help='use gpu?')
parser.add_argument('--no_logging', type=bool, default=False,
                            help="are you logging this experiment?")
parser.add_argument('--log_dir', type=str, default="/external/cnet/checkpoints",
                            help="are you logging this experiment?")
parser.add_argument('--exp_dir', type=str, default='test',
                            help='name of experiment, chkpts stored in checkpoints/experiment')
parser.add_argument('--exp_name', type=str, default='test',
                            help='name of experiment, chkpts stored in checkpoints/exp_dir/exp_name')
parser.add_argument('--test_loader', type=str, default='annotated',
                            help='test dataloader, set to annotated (default) to compute accuracy, no_annotated to just run without computing accuracy')

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--nr", type=int, default=0)
parser.add_argument('--use_bn', type=bool, default=False)
parser.add_argument('--training_frac', default=0.2,
                    help='fraction of real dataset used for finetuning (we will use 1-this for testing)')


# Get input and target tensor keys
args = parser.parse_args()

def copy_state_dict(input_dict):
    output_dict = {}
    for k,v in input_dict.items():
        n = k.strip().split('.')
        if n[0] == 'module':
            n = n[1:]
        kout = '.'.join(n)
        output_dict[kout] = v
    return output_dict

def main():
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    test(0, args)

def test(gpu, args):
    print("Starting...")
    print("Using {} percent of data for testing".format(100 - args.training_frac*100))
    torch.autograd.set_detect_anomaly(True)
    #torch.manual_seed(args.torch_seed)
    #torch.cuda.manual_seed(args.cuda_seed)
    torch.cuda.set_device(gpu)
    
    DATA_ROOT = args.data_root
    NUM_IMAGES = args.num_test_images
    CHKPNT_PTH = args.checkpoint_path
    
    if args.model_name == 'dss':
        model = DSS_Net(args, n_channels=3, n_classes=1, bilinear=True)
        loss = FocalLoss()
    elif args.model_name == 'unet':
        model = UNet(args)
        loss = nn.BCELoss()
    else:
        raise NotImplementedError
    
    state_dict = torch.load(CHKPNT_PTH)
    new_state_dict = copy_state_dict(state_dict) #OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     names = name.strip().split('.')
    #     if names[1] == 'inc':
    #         names[1] = 'conv1'
    #     name = '.'.join(names)
    #     # print(names)
    #     new_state_dict[name] = v
    
    # print("Expected values:", model.state_dict().keys())
    model.load_state_dict(new_state_dict)
    model.cuda(gpu)
    model.eval()

    if args.test_loader == 'annotated':
        dataset = SmokeDataset(dataset_limit= NUM_IMAGES, training_frac=args.training_frac)
        # dataset = SyntheticSmokeTrain(args,dataset_limit=50)
    else:
        dataset = SimpleSmokeVal(args = args,data_root = DATA_ROOT, dataset_limit = NUM_IMAGES)
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=4, pin_memory=True)#, sampler=train_sampler)
        
    # if not args.no_logging and gpu == 1:
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    log_dir = os.path.join(args.log_dir, args.exp_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if args.exp_name == "":
        exp_name = datetime.datetime.now().strftime("%H%M%S-%Y%m%d")
    else:
        exp_name = args.exp_name
        log_dir = os.path.join(log_dir, exp_name)
    writer = SummaryWriter(log_dir)
    
    iou_sum = 0
    iou_count = 0
    iou_ = 0
    for idx, data in enumerate(dataloader):
        if args.test_loader == 'annotated':
            out_img, iou_ = val_step_with_loss(data, model)
            iou_sum += iou_
            iou_count += 1
            writer.add_images('true_mask', data['target_mask']>0, idx)
        else:
            out_img = val_step(data, model)
        writer.add_images('input_img', data['input_img'], idx)
        writer.add_images('pred_mask',out_img, idx)
        writer.add_scalar(f'accuracy/test', iou_, idx)
        writer.flush()
        # print("Step: {}/{}: IOU: {}".format(idx,len(dataloader), iou_))
        if idx > len(dataloader):
            break
    if iou_count > 0:
        iou = iou_sum / iou_count
        writer.add_scalar(f'mean_accuracy/test', iou)
        print("Mean IOU: ", iou)
    print("Done")
def val_step(data_dict, model):

    # Possibly transfer to Cuda
    if args.cuda:
        data_dict['input_img'] = data_dict['input_img'].cuda(non_blocking=True)

    with torch.no_grad():
        pred_mask = model(data_dict['input_img'])
        pred_mask = pred_mask > 0.5
    return pred_mask

def val_step_with_loss(data_dict, model):
    iou_val = 0
    # Possibly transfer to Cuda
    if args.cuda:
        data_dict['input_img'] = data_dict['input_img'].cuda(non_blocking=True)
        data_dict['target_mask'] = data_dict['target_mask'].cuda(non_blocking=True)

    with torch.no_grad():
        print("Shape of input to model: ", data_dict['input_img'].shape)
        pred_mask = model(data_dict['input_img']) > 0.5
        gt_mask = data_dict['target_mask'] > 0
        iou_val = iou_pytorch(pred_mask, gt_mask)
    return pred_mask, iou_val

if __name__ == '__main__':
    main()
