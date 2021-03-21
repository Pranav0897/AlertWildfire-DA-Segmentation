import os
import argparse

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from data_gen import data_transforms

# IMG_FOLDER = 'data/alphamatting/input_lowres'
# TRIMAP_FOLDERS = ['data/alphamatting/trimap_lowres/Trimap2']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_FOLDER = 'data/trial/a'
TRIMAP_FOLDERS = ['data/trial/b']
OUTPUT_FOLDERS = ['images/trial_output']

def composite4(fg, bg, a, w, h):
    print(fg.shape, bg.shape, a.shape, w, h)
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", "-i"),
    parser.add_argument("--trimap_folder",  "-t"),
    parser.add_argument("--output_folder", "-o")
    parser.add_argument("--new_bg_img", "-b")
    args = parser.parse_args()
    
    IMG_FOLDER = args.img_folder
    TRIMAP_FOLDER = args.trimap_folder
    OUTPUT_FOLDER = args.output_folder
    
    
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']


    files = [f for f in os.listdir(IMG_FOLDER) if f.endswith('.jpg')]

    for file in tqdm(files):
        filename = os.path.join(IMG_FOLDER, file)
        img = cv.imread(filename)
#         print(img.shape)
        h, w = img.shape[:2]

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        image = img[..., ::-1]  # RGB
        image = transforms.ToPILImage()(image)
        image = transformer(image)
        x[0:, 0:3, :, :] = image

        
        filename = os.path.join(TRIMAP_FOLDER, file)
        print('reading {}...'.format(filename))
        trimap = cv.imread(filename, 0)
        x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.)
        

        # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            pred = model(x)

        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))

        pred[trimap == 0] = 0.0
        pred[trimap == 255] = 1.0

        out = (pred.copy() * 255).astype(np.uint8)

        filename = os.path.join(OUTPUT_FOLDER, file)
        cv.imwrite(filename, out)
#         print('wrote {}.'.format(filename))

#         bg_test = 'data/bg_test'
#         new_bg = 'Axis-ButtLake_2949.jpg'
#         new_bg = cv.imread(os.path.join(bg_test, new_bg))
        new_bg = cv.imread(args.new_bg_img)
        bh, bw = new_bg.shape[:2]
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio
#         print('ratio: ' + str(ratio))
        if ratio > 1:
            new_bg = cv.resize(src=new_bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)),
                               interpolation=cv.INTER_CUBIC)

        im, bg = composite4(img, new_bg, pred, w, h)
        cv.imwrite(os.path.join(OUTPUT_FOLDER,'{}_compose.png'.format(file)), im)

