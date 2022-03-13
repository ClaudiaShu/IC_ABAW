import cv2
import numpy as np
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from utils import get_lmks_by_img, get_model_by_name, get_preds, decode_preds, crop
from utils import show_landmarks, set_circles_on_img
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

def write_img(path, img):
    vutils.save_image(img, path)

# AFLW COFW WFLW
model = get_model_by_name('COFW', device='cuda')
# Read image
filename = '/data/users/ys221/data/ABAW/origin_faces/24-30-1920x1080-1/00022.jpg'
# file = 'D:/Self Improvement/Study/MRes IC/!Personal project/ys221/software/data/origin_faces/2-30-640x360/00014.jpg'
img = cv2.imread(filename, 1)
# img = Image.open(filename).convert('RGB')
h, w, c = img.shape

lmks = get_lmks_by_img(model, img)
image = np.zeros([h, w])
# print(lmks)
show_landmarks(image, lmks)
temp_img = set_circles_on_img(image, lmks, circle_size=1, color=(255,255,255), is_copy=False)
cv2.imwrite("/data/users/ys221/data/results/landmark.jpg", temp_img)

# fake code
# 1. glob {clip}/{name}
# 2. read image
# 3. get landmark
# 4. save



