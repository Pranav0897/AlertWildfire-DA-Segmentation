import argparse
import datetime
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from my_dataset import SyntheticSmokeTrain, SimpleSmokeTrain, SimpleSmokeVal, SmokeDataset

from compute_iou import iou_pytorch

from collections import OrderedDict, Counter
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    ### for running different versions of pytorch
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    ###
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    log_dir = args.save
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    exp_name = datetime.datetime.now().strftime("%H%M%S-%Y%m%d")
    log_dir = os.path.join(log_dir, exp_name)
    writer = SummaryWriter(log_dir)

    # testloader = data.DataLoader(SyntheticSmokeTrain(args={}, dataset_limit=-1, #args.num_steps * args.iter_size * args.batch_size,
    #                 image_shape=(360,640), dataset_mean=IMG_MEAN),
    #                     batch_size=1, shuffle=True, pin_memory=True)

    testloader = data.DataLoader(SmokeDataset(image_size=(640,360), dataset_mean=IMG_MEAN),
                        batch_size=1, shuffle=True, pin_memory=True)
    # testloader = data.DataLoader(SimpleSmokeTrain(args = {}, image_size=(640,360), dataset_mean=IMG_MEAN),
    #                     batch_size=1, shuffle=True, pin_memory=True)
    # testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    # batch_size=1, shuffle=False, pin_memory=True)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(640,360), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(640,360), mode='bilinear', align_corners=True)

    count = 0
    iou_sum_fg = 0
    iou_count_fg = 0

    iou_sum_bg = 0
    iou_count_bg = 0

    for index, batch in enumerate(testloader):
        if (index+1) % 100 == 0:
            print('%d processd' % index)
            # print("Processed {}/{}".format(index, len(testloader)))

        # if count > 5:
        #     break
        image, label, name = batch
        if args.model == 'DeeplabMulti':
            with torch.no_grad():
                output1, output2 = model(Variable(image).cuda(gpu0))
            # print(output1.shape)
            # print(output2.shape)
            output = interp(output2).cpu()
            orig_output = output.detach().clone()
            output = output.data[0].numpy()
            # output = (output > 0.5).astype(np.uint8)*255
            # print(np.all(output==0), np.all(output==255))
            # print(np.min(output), np.max(output))

        elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
            with torch.no_grad():
                output = model(Variable(image).cuda(gpu0))
            output = interp(output).cpu().data[0].numpy()

        
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        classes_seen = set(output.ravel().tolist())
        # print(classes_seen)
        # print(output.shape, name[0])
        output_col = colorize_mask(output)
        output = Image.fromarray(output)
        # print("name", name)
        name = name[0]
        # name = name[0].split('/')[-1]
        
        if len(classes_seen) > 1:
            count += 1
            print(classes_seen)
            print(Counter(np.asarray(output).ravel()))
            image = image.squeeze()
            for c in range(3):
                image[c,:,:] += IMG_MEAN[c]
                # image2[c,:,:] += IMG_MEAN[2-c]
            image = (image - image.min())/(image.max()-image.min())
            image = image[[2,1,0],:,:]
            print(image.shape, image.min(), image.max())
            output.save(os.path.join(args.save, name+'.png'))
            output_col.save(os.path.join(args.save, name+'_color.png'))
            # output.save('%s/%s.png' % (args.save, name))
            # output_col.save('%s/%s_color.png' % (args.save, name))#.split('.')[0]))

            output_argmaxs = torch.argmax(orig_output.squeeze(), dim=0)
            mask1 = (output_argmaxs == 0).float()*255
            label = label.squeeze()

            iou_fg = iou_pytorch(mask1, label)
            print("foreground IoU", iou_fg)
            iou_sum_fg += iou_fg
            iou_count_fg += 1
            

            mask2 = (output_argmaxs > 0).float()*255
            label2 = label.max() - label

            iou_bg = iou_pytorch(mask2, label2)
            print("IoU for background: ", iou_bg)
            iou_sum_bg += iou_bg
            iou_count_bg += 1


            writer.add_images(f'input_images',tf.resize(image[[2,1,0]],[1080,1920]), index, dataformats='CHW')

            print("shape of label", label.shape)
            label_reshaped = tf.resize(label.unsqueeze(0), [1080,1920]).squeeze()
            print("label reshaped: ", label_reshaped.shape)
            writer.add_images(f'labels', label_reshaped, index, dataformats='HW')
            writer.add_images(f'output/1',255- np.asarray(tf.resize(output, [1080,1920]))*255, index,dataformats='HW')
            # writer.add_images(f'output/1',np.asarray(output)*255, index,dataformats='HW')
            # writer.add_images(f'output/2',np.asarray(output_col), index, dataformats='HW')
            writer.add_scalar(f'iou/smoke', iou_fg, index)
            writer.add_scalar(f'iou/background', iou_bg, index)
            writer.add_scalar(f'iou/mean', (iou_bg+iou_fg)/2, index)
            writer.flush()

    if iou_count_fg > 0:
        print("Mean IoU, foreground: {}".format(iou_sum_fg/iou_count_fg))
        print("Mean IoU, background: {}".format(iou_sum_bg/iou_count_bg))
        print("Mean IoU, averaged over classes: {}".format((iou_sum_fg+iou_sum_bg)/(iou_count_fg+iou_count_bg)))

if __name__ == '__main__':
    main()
