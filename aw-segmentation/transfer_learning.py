import os
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
# from augmentations import SyntheticAugmentation
from SyntheticSmokeDataset import SyntheticSmokeTrain, SmokeDataset, SimpleSmokeTrain

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

# distributed params
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--nr", type=int, default=0)

# runtime params
parser.add_argument('--data_root', help='path to train imgs', required=True)
parser.add_argument('--overlays_root', help='path to train imgs', required=True)
parser.add_argument('--val_data_root', help='path to val imgs', required=True)
parser.add_argument('--val_anns_root', help='path to val imgs', required=True)
parser.add_argument('--epochs', type=int, required=True,
                    help='number of epochs to run')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from checkpoint (using experiment name)')
parser.add_argument('--cuda', type=bool, default=True, help='use gpu?')
parser.add_argument('--no_logging', type=bool, default=False,
                    help="are you logging this experiment?")
parser.add_argument('--log_dir', type=str, default="/external/cnet/checkpoints",
                    help="are you logging this experiment?")
parser.add_argument('--log_freq', type=int, default=1, help='how often to log statistics')
parser.add_argument('--save_freq', type=int, default=1, help='how often to save model')
parser.add_argument('--exp_dir', type=str, default='test',
                    help='name of experiment, chkpts stored in checkpoints/experiment')
parser.add_argument('--exp_name', type=str, default='test',
                    help='name of experiment, chkpts stored in checkpoints/exp_dir/exp_name')
parser.add_argument('--validate', type=bool, default=False,
                    help='set to true if validating model')
parser.add_argument('--ckpt', type=str, default="",
                    help="path to model checkpoint if using one")
parser.add_argument('--use_pretrained', type=bool,
                    default=False, help="use pretrained model from authors")

# module params
parser.add_argument('--model_name', type=str,
                    default="dss", help="name of model")

# dataset params
parser.add_argument('--dataset_name', default='synthetic', help='synthetic or annotated')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')

parser.add_argument('--num_examples', type=int, default=-1,
                    help="number of examples to train on per epoch")
parser.add_argument('--num_workers', type=int, default=8,
                    help="number of workers for the dataloader")
parser.add_argument('--shuffle_dataset', type=bool,
                    default=False, help='shuffle the dataset?')

# learning params
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate')
parser.add_argument('--lr_sched_type', type=str, default="none",
                    help="path to model checkpoint if using one")
parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for sgd or alpha param for adam')
parser.add_argument('--beta', type=float, default=0.999,
                    help='beta param for adam')
parser.add_argument('--weight_decay', type=float,
                    default=0.0, help='weight decay')
parser.add_argument('--dropout', type=bool, default=False,
                    help='dropout for regularization', choices=[True, False])
parser.add_argument('--grad_clip', type=float, default=0,
                    help='gradient clipping threshold')

# model params
parser.add_argument('--use_bn', type=bool, default=False,
                    help="whether to use batch-norm in training procedure")

# etc.
parser.add_argument('--multi_gpu', type=bool,
                    default=False, help='use multiple gpus')
parser.add_argument('--num_gpus', type=int, default=2)
parser.add_argument('--debugging', type=bool,
                    default=False, help='are you debugging?')
parser.add_argument('--finetuning', type=bool, default=False,
                    help='finetuning on supervised data')
parser.add_argument('--evaluation', type=bool,
                    default=False, help='evaluating on data')
parser.add_argument('--torch_seed', default=123768,
                    help='random seed for reproducibility')
parser.add_argument('--cuda_seed', default=543987,
                    help='random seed for reproducibility')
parser.add_argument('--training_frac', default=0.2,
                    help='fraction of real dataset to use for finetuning')


args = parser.parse_args()


def main():
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    train(args.num_gpus-1, args)
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '8888'
    # args.world_size = args.num_gpus * args.num_nodes
    # mp.spawn(train, nprocs=args.num_gpus, args=(args,))

def copy_state_dict(input_dict):
    output_dict = {}
    for k,v in input_dict.items():
        n = k.strip().split('.')
        if n[0] == 'module':
            n = n[1:]
        kout = '.'.join(n)
        output_dict[kout] = v
    return output_dict

def train(gpu, args):
    
    # rank = args.nr * args.num_gpus + gpu

    # dist.init_process_group(backend="nccl", world_size=args.world_size, rank=rank)

    print("Using {} percent of the target data for finetuning".format(args.training_frac*100))
    
    if args.batch_size == 1 and args.use_bn is True:
        raise Exception

    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed(args.cuda_seed)
    
    torch.cuda.set_device(gpu)
    
    DATASET_NAME = args.dataset_name
    DATA_ROOT = args.data_root
    OVERLAYS_ROOT = args.overlays_root

    if args.model_name == 'dss':
        model = DSS_Net(args, n_channels=3, n_classes=1, bilinear=True)
        loss = FocalLoss()
    elif args.model_name == 'unet':
        model = UNet(args)
        loss = nn.BCELoss()
    else:
        raise NotImplementedError

    #model = nn.SyncBatchNorm(model)

    print(f"Using {torch.cuda.device_count()} GPUs...")
        
    # define dataset
    if DATASET_NAME == 'synthetic':
        assert (args.overlays_root != "")
        train_dataset = SmokeDataset(dataset_limit=args.num_examples, training = True, training_frac=args.training_frac)
        # train_dataset = SyntheticSmokeTrain(args, DATA_ROOT, OVERLAYS_ROOT, dataset_limit=args.num_examples)
        # train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)#, sampler=train_sampler)
        if args.validate:
            val_dataset = SmokeDataset()
        else:
            val_dataset = None
        val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None
    else:
        raise NotImplementedError

    # define augmentations
    augmentations = None #SyntheticAugmentation(args)
    
    # load the model
    print("Loding model and augmentations and placing on gpu...")

    if args.cuda:
        if augmentations is not None:
            augmentations = augmentations.cuda()

        model = model.cuda(device=gpu)
            
        # if args.num_gpus > 0 or torch.cuda.device_count() > 0:
        #     model = DistributedDataParallel(model, device_ids=[gpu])
                
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} learnable parameters")

    # load optimizer and lr scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, betas=[args.momentum, args.beta], weight_decay=args.weight_decay)

    if args.lr_sched_type == 'plateau':
        print("Using plateau lr schedule")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_gamma, verbose=True, mode='min', patience=10)
    elif args.lr_sched_type == 'step':
        print("Using step lr schedule")
        milestones = [30]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.lr_gamma)
    elif args.lr_sched_type == 'none':
        lr_scheduler = None

    # set up logging
    # if not args.no_logging and gpu == 0:
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

    if args.ckpt != "" and args.use_pretrained:
        state_dict = torch.load(args.ckpt)#['state_dict']
        model.load_state_dict(copy_state_dict(state_dict))
    elif args.start_epoch > 0:
        load_epoch = args.start_epoch - 1
        ckpt_fp = os.path.join(log_dir, f"{load_epoch}.ckpt")

        print(f"Loading model from {ckpt_fp}...")

        ckpt = torch.load(ckpt_fp)
        assert (ckpt['epoch'] ==
                load_epoch), "epoch from state dict does not match with args"
        model.load_state_dict(ckpt)

    model.train()
    
    # run training loop
    for epoch in range(args.start_epoch, args.epochs + 1):
        print(f"Training epoch: {epoch}...")
        # train_sampler.set_epoch(epoch)
        freeze_layers = epoch < 10
        train_loss_avg, pred_mask, input_dict = train_one_epoch(
            args, model, loss, train_dataloader, optimizer, augmentations, lr_scheduler, freeze_layers)
        if gpu == 0:
            print(f"\t Epoch {epoch} train loss avg:")
            pprint(train_loss_avg)

        if val_dataset is not None:
            print(f"Validation epoch: {epoch}...")
            val_loss_avg = eval(args, model, loss, val_dataloader, augmentations)
            print(f"\t Epoch {epoch} val loss avg: {val_loss_avg}")

        if not args.no_logging and gpu == 0:
            writer.add_scalar(f'loss/train', train_loss_avg, epoch)
            if epoch % args.log_freq == 0:
                visualize_output(args, input_dict, pred_mask, epoch, writer)

        if args.lr_sched_type == 'plateau':
            lr_scheduler.step(train_loss_avg_dict['total_loss'])
        elif args.lr_sched_type == 'step':
            lr_scheduler.step(epoch)

        # save model
        if not args.no_logging:
            if epoch % args.save_freq == 0 or epoch == args.epochs:
                fp = os.path.join(log_dir, f"finetune_{epoch}.ckpt")
                print("saving model to: ",fp)
                torch.save(model.state_dict(), fp)

            writer.flush()

    return


def step(args, data_dict, model, loss, augmentations):
    # Get input and target tensor keys
    input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
    target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
    tensor_keys = input_keys + target_keys

    # Possibly transfer to Cuda
    if args.cuda:
        for k, v in data_dict.items():
            if k in tensor_keys:
                data_dict[k] = v.cuda(non_blocking=True)

    if augmentations is not None:
        with torch.no_grad():
            data_dict = augmentations(data_dict)

    for k, t in data_dict.items():
        if k in input_keys:
            data_dict[k] = t.requires_grad_(True)
        if k in target_keys:
            data_dict[k] = t.requires_grad_(False)

    pred_mask = model(data_dict['input_img'])
    #print("pred_mask.shape: ", pred_mask.shape)
    #print("data_dict['target_mask'].shape: ", data_dict['target_mask'].shape)
    total_loss = loss(pred_mask, data_dict['target_mask'])

    return total_loss, pred_mask 

# def write_one_epoch(args, dataloader, augmentations, xpath, ypath):
#     # Get input and target tensor keys
#     ld = len(dataloader)
#     for idx, data_dict in enumerate(dataloader):
#         data_dict = augmentations(data_dict)
#         cv2.imwrite(os.path.join(xpath, str(idx)+".png"), data_dict['input_img'])
#         cv2.imwrite(os.path.join(ypath, str(idx)+".png"), data_dict['target_mask'])
#         print("{}/{}\r".format(idx, ld), end = "")

def train_one_epoch(args, model, loss, dataloader, optimizer, augmentations, lr_scheduler, freeze_layers=True):

    total_loss = 0
    if freeze_layers:
        for layers in model.encoder.res_layers[-2:]:
            for param in layers.parameters():
                param.requires_grad = False
        for layers in model.decoder.convs[-4:]:
            for param in layers.parameters():
                param.requires_grad = False
    else:
        for params in model.parameters():
            params.requires_grad = True

    for data in tqdm(dataloader):
        step_loss, pred_mask = step(
            args, data, model, loss, augmentations)
        
#         info = nvmlDeviceGetMemoryInfo(h)
#         print(f'total    : {info.total / (1024**3)} Gb')
#         print(f'free     : {info.free / (1024**3)} Gb')
#         print(f'used     : {info.used / (1024**3)} Gb')

        total_loss += step_loss.item()
        
        # calculate gradients and then do Adam step
        optimizer.zero_grad()
        step_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()

    loss_avg = total_loss / len(dataloader)

    return loss_avg, pred_mask.detach(), data


def eval(args, model, loss, dataloader, augmentations):
    torch.cuda.empty_cache()
    model.eval()
    val_loss_avg = 0
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            step_loss, pred_mask = step(
                args, data, model, loss, augmentations)
            total_loss += step_loss.item()
        val_loss_avg = total_loss / len(dataloader)
    model.train()
    torch.cuda.empty_cache()
    return val_loss_avg


def visualize_output(args, input_dict, pred_mask, epoch, writer):

    assert (writer is not None), "tensorboard writer not provided"

    input_img = input_dict['input_img'].detach()
    target_mask = input_dict['target_mask'].detach()

    # input imgs
    writer.add_images('input_img', input_img, epoch)
    writer.add_images('target_mask', target_mask, epoch)
    writer.add_images('pred_mask', pred_mask, epoch)

if __name__ == '__main__':
    main()
