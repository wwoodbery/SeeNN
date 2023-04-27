import numpy as np
from skimage import io, img_as_ubyte, img_as_float32, util, img_as_uint
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from skimage import filters
import cv2
import pickle
import os

P_CROP_DIM = 364
C_CROP_SIZE = 4
HR_DIM = P_CROP_DIM - C_CROP_SIZE
DSCALE_FACTOR = 4
LR_DIM = HR_DIM // DSCALE_FACTOR


def crop_center(img):
    start = P_CROP_DIM//2-(HR_DIM//2)
    return img[start:start+HR_DIM,start:start+HR_DIM]

def create_hr(img, crop=True):
    img = img[:,:,:3]  # ensure all images have 3 color channels
    if crop:
        return crop_center(resize(img, (P_CROP_DIM, P_CROP_DIM, 3), anti_aliasing=True))
    else:
        return resize(img, (P_CROP_DIM, P_CROP_DIM, 3), anti_aliasing=True)

def create_lr(hr_img, interp='bicubic'):
    if interp == 'bicubic':
        return cv2.resize(filters.gaussian(hr_img), (LR_DIM, LR_DIM), interpolation=cv2.INTER_CUBIC)
    else:
        return downscale_local_mean(filters.gaussian(hr_img), (DSCALE_FACTOR, DSCALE_FACTOR, 1))

def generate_lr_hr_list(list_of_paths):
    lr_hr_tuples = []
    for path in list_of_paths:
        image = io.imread(path)
        if len(image.shape) == 3:
            image = create_hr(image, True)
            lr = create_lr(image, interp='bicubic')
            lr_hr_tuples.append((lr, image))
            if len(lr_hr_tuples) == 1000:
                break
    return lr_hr_tuples

def generate_data_file(tuple_list):
    with open('lr_hr_image_data2.pkl', 'wb') as f:
        pickle.dump(tuple_list, f)
        f.close()

def get_data_from_file(file_path):
    with open(file_path, 'rb') as f:
        lr_hr_tuples = pickle.load(f)
        f.close()
    return lr_hr_tuples

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        cnt = 0
        for name in files:
            r.append(os.path.join(root, name))
            cnt+=1
            if cnt >= 3:
              break
    return r

r = list_files('val')
print(len(r))
lr_hr_tuples = generate_lr_hr_list(r)
print(len(lr_hr_tuples))
generate_data_file(lr_hr_tuples)