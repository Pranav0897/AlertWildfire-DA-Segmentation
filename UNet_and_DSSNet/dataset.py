import os
import json
import torch
import random
import numpy as np
from random import randint

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import matplotlib.path as path
from matplotlib.path import Path
from scipy.signal import convolve2d
from scipy.stats import iqr

def norm(img):
    return (img-img.min())/(img.max()-img.min())

class SyntheticSmokeTrain(Dataset):
    def __init__(self, 
                 args,
                 data_root="/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/non-smoke/", 
                 overlays_root="/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/synth_smoke/masks/",
                 mask_thresh= 0.3,
                 dataset_limit = -1,
                 debug_train = False):
        
        self.args = args
        self.debug_train = debug_train
        self.dataset_limit = dataset_limit
        self.data_root = data_root
        self.overlays_root = overlays_root
        self.img_fns = []
        self.mask_thresh = mask_thresh
        self.camera_sample_rate = 10
        
        data = open('../aw-segmentation-old/daytime_images_list.txt','r').read().strip().split('\n')
        data = data + open('../aw-segmentation-old/daytime_images_list1.txt','r').read().strip().split('\n')
        
        data = random.sample(list(set(data)), len(data)//self.camera_sample_rate)
        
        self.img_fns = data
        self.img_fns = self.img_fns[:self.dataset_limit]
        
        self.overlay_fns = [os.path.join(overlays_root, file) for file in os.listdir(overlays_root)]
        self.n_overlays = len(os.listdir(self.overlays_root))
        self.rn = randint(0, self.n_overlays-1)
        self.box_filter = 1/9 * np.eye(3)
        self.gamma = 1.5

        print("Debug mode: {}, dataset_limit: {}".format(self.debug_train, self.dataset_limit))

    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, idx):
        idx = idx % len(self.img_fns)
        
        img_fn = self.img_fns[idx]
        img = Image.open(img_fn)
        img = img.resize((1024,512))
        width, height = img.size
        gray = np.asarray(img.convert('L'))
        img = transforms.functional.to_tensor(img)
        
        p_neg = np.random.rand() > 0.05
        
        if p_neg:
            if self.debug_train:
                overlay_fn = self.overlay_fns[self.rn]
            else:
                overlay_fn = self.overlay_fns[randint(0, self.n_overlays-1)]
            overlay_img = Image.open(overlay_fn).convert('L')

            spatial_augs = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=60, translate=(.3, .1), scale=(.35, .7), fillcolor=(255)),
                transforms.Resize((height, width)),
            ] if not self.debug_train else [transforms.Resize((height, width))])

            overlay_img = spatial_augs(overlay_img)
            overlay_img = 1.0 - transforms.functional.to_tensor(overlay_img)

            # create mask according to iqr and add color jitter
            iqr_val = iqr(overlay_img[overlay_img > 0])
            mask = (overlay_img <= iqr_val).squeeze().float()

            overlay_img = transforms.functional.to_pil_image(overlay_img)

            cj = transforms.ColorJitter(brightness=(0.8, 1.2), saturation=(0.8, 1.0))
            overlay_img = cj(overlay_img)

            mean_gray = norm(convolve2d(gray, self.box_filter,'same'))
            overlay_np = np.asarray(overlay_img)
            final_img= np.power(np.multiply(mean_gray, overlay_np), self.gamma)
            #mask = norm(np.where(overlay_np > 200, final_img, 0))
            #mask = transforms.functional.to_tensor(mask).float()
            img[:, mask[0]>0] = mask[0,mask[0]>0]
            #mask = torch.where(mask>0,torch.ones_like(mask),torch.zeros_like(mask))
            # overlay_img = transforms.functional.to_tensor(overlay_img)
            
            # img[:, mask==1] = overlay_img[0, mask==1]
        else:
            mask = torch.zeros_like(img[0, :, :])
            
        img_spatial_augs = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),
            transforms.CenterCrop((height-40, width-60)),
            transforms.Resize((height,width))
        ] if not self.debug_train else [transforms.ToPILImage(), transforms.CenterCrop((height-40, width-60)), transforms.Resize((height, width))])

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed=seed)
        
        img = img_spatial_augs(img)
        random.seed(seed)
        torch.manual_seed(seed=seed)
        mask = img_spatial_augs(mask)

        img_color_augs = transforms.Compose([
            transforms.ColorJitter(brightness=(0.7, 1.3), saturation=(0.7, 1.2), contrast=(0.8, 1.2)),
            transforms.ToTensor()
        ] if not self.debug_train else [transforms.ToTensor()])

        img = img_color_augs(img)
        mask = transforms.functional.to_tensor(mask)

        output_dict = {
            "idx": idx,
            "input_img": img,
            "target_mask": mask
        } 
        return output_dict


class SmokeDataset(Dataset):
    def __init__(self, 
                 data_root="/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/New dataset/", 
                 anns_root="/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/New Dataset Annotations/",
                 dataset_limit=-1):
        
        self.dataset_limit = dataset_limit
        self.data_root = data_root
        self.anns_root = anns_root
        img_fns = list(sorted(os.listdir('/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/New dataset'), key=lambda x: int(x[:-4])))
        img_fns = [fn for fn in img_fns]
        ann_fns = [fn[:-4]+'.json' for fn in img_fns]
               
        self.img_fns = img_fns[:self.dataset_limit]
        self.ann_fns = ann_fns[:self.dataset_limit]
        
    def __len__(self):
        return len(self.img_fns)
    
    def __getitem__(self, idx):
        
        idx = idx % len(self.img_fns)
        
        img_fn = self.img_fns[idx]
        ann_fn = self.ann_fns[idx]
        
        img = Image.open(self.data_root + img_fn)
        
        with open(self.anns_root + ann_fn) as f:
            data_dict = {}
            annotations = json.load(f)
            for annotation in annotations:
                h, w = 1080, 1920
                mask_attrs = list(filter(lambda x: x['region_attributes']['name'].strip() in ['smoke', 'smoke1', ''], annotation['regions']))[0]['shape_attributes']
                pts = np.array(list(zip(mask_attrs['all_points_x'], mask_attrs['all_points_y'])))
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                x, y = x.flatten(), y.flatten()
                meshgrid = np.vstack((x,y)).T
                path = Path(pts)
                mask = path.contains_points(meshgrid)
                mask = mask.reshape((h,w)).astype(np.float32)
                data_dict['mask'] = mask
            assert (data_dict['mask'].sum() > 0), "No points in mask"
        
        mask = tf.to_pil_image(mask)
        mask = tf.resize(mask, (512, 1024))
        mask = tf.to_tensor(mask)
        img = tf.resize(img, (512, 1024))
        img = tf.to_tensor(img)
        
        output_dict = {
            "idx": idx,
            "input_img": img,
            "target_mask": mask
        } 
        return output_dict
