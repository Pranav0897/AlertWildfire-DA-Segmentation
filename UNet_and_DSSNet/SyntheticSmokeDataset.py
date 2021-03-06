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
from pathlib import Path as PathDir
from scipy.signal import convolve2d

from scipy.stats import iqr

def norm(img):
    return (img-img.min())/(img.max()-img.min())

class SimpleSmokeTrain(Dataset):
    def __init__(self,
                 args,
                 data_root = '/data/field/nextcloud_nautilus_sync/AlertWildfire/redis_nonsmoke_dataset/data/',
                 mask_root = '/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/synth_dataset/masks/',
                 dataset_limit = -1,
                 image_size = (832,512),
                 dataset_mean = (0.5,0.5,0.5)):

        self.args = args
        self.dataset_limit = dataset_limit
        self.data_root = data_root
        self.mask_root = mask_root
        self.image_size = image_size        
        self.datapoints = list(os.listdir(self.data_root))[:self.dataset_limit]
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, idx):
        idx = idx % len(self.datapoints)
        img_ = Image.open(
                os.path.join(
                    self.data_root, self.datapoints[idx]
                )
            ).resize(self.image_size)
        if self.mask_root is not None:
            mask_ = Image.open(
                    os.path.join(
                        self.mask_root, self.datapoints[idx]
                    )
                ).convert('L').resize(self.image_size)
            mask = transforms.functional.to_tensor(mask_)
        else:
            mask = None
        img = transforms.functional.to_tensor(img_)
        output_dict = {
            "idx": idx,
            "input_img": img,
            "target_mask": mask
        } 
        return output_dict

class SimpleSmokeVal(Dataset):
    def __init__(self,
                args,
                data_root='/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/smoke/',
                dataset_limit=-1):

        self.args = args
        self.data_root = data_root
        self.data_point = list(os.listdir(data_root))[:dataset_limit]
        random.shuffle(self.data_point)

    def __len__(self):
        return len(self.data_point)
    
    def __getitem__(self, idx):
        idx = idx % len(self.data_point)
        img_ = Image.open(
                os.path.join(
                    self.data_root, self.data_point[idx]
                    )
                ).resize((832,512))
        img = transforms.functional.to_tensor(img_)
        output_dict = {
                "idx": idx,
                "name":self.data_point[idx],
                "input_img":img
        }
        return output_dict

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
        self.camera_sample_rate = 5
        
        #data = open('./daytime_images_list.txt','r').read().strip().split('\n')
        #data = data + open('./daytime_images_list1.txt','r').read().strip().split('\n')
        
        data = [str(i) for i in PathDir(data_root).rglob('*.jpg')]
        data = random.sample(list(set(data)), len(data)//self.camera_sample_rate)
        
        self.img_fns = data
        self.img_fns = self.img_fns[:self.dataset_limit]
        
        self.overlay_fns = [os.path.join(overlays_root, file) for file in os.listdir(overlays_root)]
        self.n_overlays = len(os.listdir(self.overlays_root))
        self.rn = randint(0, self.n_overlays-1)
        self.box_filter = 1/9 * np.eye(3)
        self.gamma = 1.5
        self.img_fn_len = len(self.img_fns)
        print("Number of non smoke images : {}, Sample path: {}".format(self.img_fn_len, self.img_fns[0]))
        print("Debug mode: {}, dataset_limit: {}".format(self.debug_train, self.dataset_limit))

    def __len__(self):
        return self.img_fn_len

    def __getitem__(self, idx):
        idx = idx % self.img_fn_len
        
        #img_fn = self.img_fns[idx]
        #img = Image.open(img_fn)
        bad_img_counts = 0
        while True:
            try:
                img_fn = self.img_fns[idx]
                img = Image.open(img_fn)
                img_np = np.asarray(img)
                if img_np is None or np.prod(list(img_np.shape)) == 0:
                    idx += 1
                    idx = idx % self.img_fn_len
                    bad_img_counts += 1
                    print("Bad data1: {}\n".format(img_fn))
                    continue
                if bad_img_counts > 100:
                    break

                break
            except:
               idx += 1
               bad_img_counts += 1
               idx = idx % self.img_fn_len
               print("Bad data2: {}\n".format(img_fn))
               if bad_img_counts > 100:
                   break

        img = img.resize((1024,512))
        width, height = img.size
        gray = np.asarray(img.convert('L'))
        img = transforms.functional.to_tensor(img)
        bg_image = img.detach().clone()

        
        p_neg = np.random.rand() > 0.05
        og_overlay_image = None
        if p_neg:
            if self.debug_train:
                overlay_fn = self.overlay_fns[self.rn]
            else:
                overlay_fn = self.overlay_fns[randint(0, self.n_overlays-1)]
            overlay_img = Image.open(overlay_fn).convert('L')
            og_overlay_image = transforms.functional.to_tensor(overlay_img)
            spatial_augs = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=60, translate=(.3, .1), scale=(.25, .5), fillcolor=(255)),
                transforms.Resize((height, width)),
            ] if not self.debug_train else [transforms.Resize((height, width))])

            overlay_img = spatial_augs(overlay_img)
            overlay_img = 1.0 - transforms.functional.to_tensor(overlay_img)

            # create mask according to iqr and add color jitter
            iqr_val = iqr(overlay_img[overlay_img > 0])
            mask = (overlay_img >= iqr_val).squeeze().float()

            overlay_img = transforms.functional.to_pil_image(overlay_img)

            # cj = transforms.ColorJitter(brightness=(0.8, 1.2), saturation=(0.8, 1.0))
            # overlay_img = cj(overlay_img)
            
            mean_gray = norm(gray)
            
            spatial_augs2 = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

            ### DEBUG
            overlay_img = spatial_augs2(overlay_img).squeeze().numpy()
            mult = np.power(mean_gray, 0.2) #np.ones_like(mean_gray) 
            final_img2 = transforms.functional.to_tensor(np.multiply(mult, overlay_img)).float()
            ### DEBUG
            
            img[:, mask==1] = final_img2[0, mask==1]
        else:
            mask = torch.zeros_like(img[0, :, :])
            
        img_spatial_augs = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),
            transforms.CenterCrop((height-40, width-60)),
            transforms.Resize((height, width)),
        ] if not self.debug_train else [transforms.CenterCrop((height-40, width-60)),
                                        transforms.Resize((height, width))])

        # use same random seed for both augmentations
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
        #if img is None or mask is None or og_overlay_image is None or idx is None:
            #print("image is None or mask is none")
        output_dict = {
            "idx": idx,
            "input_img": img,
            "target_mask": mask,
            "bg": bg_image,
            "fg": og_overlay_image
        } 
        return output_dict


class SmokeDataset(Dataset):
    def __init__(self, 
                 data_root="/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/New dataset/", 
                 anns_root="/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/New Dataset Annotations/",
                 dataset_limit=-1,
                 training = False,
                 training_frac = 0.2):
        
        self.dataset_limit = dataset_limit
        self.data_root = data_root
        self.anns_root = anns_root
        img_fns = list(sorted(os.listdir('/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/New dataset'), key=lambda x: int(x[:-4])))
        l = int(training_frac * len(img_fns))
        if training:
            img_fns = img_fns[:l]
        else:
            img_fns = img_fns[l:]

        img_fns = [fn for fn in img_fns]
        ann_fns = [fn[:-4]+'.json' for fn in img_fns]
               
        self.other_data_mean = torch.tensor([0.3986,0.4096,0.4099])
        self.img_fns = img_fns[:self.dataset_limit]
        self.ann_fns = ann_fns[:self.dataset_limit]
        print("self.data_root: ",self.data_root, type(self.data_root))
    def __len__(self):
        return len(self.img_fns)
    
    def __getitem__(self, idx):
        
        idx = idx % len(self.img_fns)
        
        img_fn = self.img_fns[idx]
        ann_fn = self.ann_fns[idx]
        
        img = Image.open(self.data_root + img_fn)
        
        with open(self.anns_root + ann_fn) as f:
            # print(ann_fn)
            data_dict = {}
            annotations = json.load(f)
            # print(len(annotations))
            for annotation in annotations:
                h, w = 1080, 1920
                mask_attrs_list = list(
                                filter(
                                    lambda x: True, #x['region_attributes']['name'].strip() == 'smoke', 
                                    annotation['regions']
                                )
                            )
                # print(mask_attrs_list)
                # print(annotation)
                mask_attrs = mask_attrs_list[0]['shape_attributes']
                
                pts = np.array(list(zip(mask_attrs['all_points_x'], mask_attrs['all_points_y'])))
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                x, y = x.flatten(), y.flatten()
                meshgrid = np.vstack((x,y)).T
                path = Path(pts)
                mask = path.contains_points(meshgrid)
                mask = mask.reshape((h,w)).astype(np.float32)
                data_dict['mask'] = mask
            assert (data_dict['mask'].sum() > 0), "No points in mask"
        
        mask = torch.from_numpy(mask)
        mask = tf.to_pil_image(mask)
        mask = tf.resize(mask, (512, 1024))
        mask = tf.to_tensor(mask)
        img = tf.resize(img, (512, 1024))
        img = tf.to_tensor(img).squeeze()
        cur_img_mean = torch.mean(img, axis=(1,2))
        for c in range(3):
            img[c] -= (self.other_data_mean[c]- cur_img_mean[c])
        
        img = img
        output_dict = {
            "idx": idx,
            "input_img": img,
            "target_mask": mask
        } 
        return output_dict
