import numpy as np
from skimage import io, img_as_ubyte, img_as_float32, util, img_as_uint
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from skimage import filters
import cv2
import pickle

P_CROP_DIM = 364
C_CROP_SIZE = 4
HR_DIM = P_CROP_DIM - C_CROP_SIZE
DSCALE_FACTOR = 4
LR_DIM = HR_DIM // DSCALE_FACTOR

def get_hr_shape():
    x = P_CROP_DIM - C_CROP_SIZE
    return (x, x, 3)

def crop_center(img):
    start = P_CROP_DIM//2-(HR_DIM//2)
    return img[start:start+HR_DIM,start:start+HR_DIM]

def create_hr(img, crop=True):
    img = img[:,:,:3]  # ensure all images have 3 color channels
    if crop:
        return crop_center(resize(img, (HR_DIM, HR_DIM, 3), anti_aliasing=True))
    else:
        return resize(img, (HR_DIM, HR_DIM, 3), anti_aliasing=True)

def create_lr(hr_img, interp='bicubic'):
    if interp == 'bicubic':
        return cv2.resize(filters.gaussian(hr_img), (LR_DIM, LR_DIM), interpolation=cv2.INTER_CUBIC)
    else:
        return downscale_local_mean(filters.gaussian(hr_img), (DSCALE_FACTOR, DSCALE_FACTOR, 1))

def generate_lr_hr_list(image_folder_path):
    lr_hr_tuples = []
    for image in io.imread_collection(image_folder_path):
        image = create_hr(image, True)
        lr = create_lr(image, interp='bicubic')
        lr_hr_tuples.append((lr, image))
    return lr_hr_tuples

def generate_data_file(tuple_list):
    with open('lr_hr_image_data.pkl', 'wb') as f:
        pickle.dump(tuple_list, f)
        f.close()

def get_data_from_file(file_path):
    with open(file_path, 'rb') as f:
        lr_hr_tuples = pickle.load(f)
        f.close()
    return lr_hr_tuples

def tuples_to_arrays(lr_hr_tuples):
    return np.array(lr_hr_tuples)

lr_hr_tuples = generate_lr_hr_list('images/*jpg')
# generate_data_file(lr_hr_tuples)

# lr_hr_tuples = get_data_from_file('lr_hr_image_data.pkl')
# for tup in lr_hr_tuples:
#     plt.imshow(tup[1])
#     plt.show()
#     plt.imshow(tup[0])
#     plt.show()
