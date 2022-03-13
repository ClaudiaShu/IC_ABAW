from models_abaw import *
from PIL import Image
import torch
from torchvision import transforms
import torchvision.utils as vutils
import glob
import cv2
import os
from models_abaw import *
from models import *
# Experimental: HDBScan is a state-of-the-art clustering algorithm
hdbscan_available = True
try:
    import hdbscan
except ImportError:
    hdbscan_available = False

# device = "cuda:0" if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
device = 'cpu'


'''
data loader
'''
# list_csv = glob.glob("/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_v2/" + '*')
# df = pd.DataFrame()
# for i in list_csv:
#     df = pd.concat((df, pd.read_csv(i)), axis=0).reset_index(drop=True)
# list_labels_ex = np.array(df['labels_ex'].values)
# list_image_id = np.array(df['image_id'].values)

model = BaseLineCycletest()
model.to(device)

filename = '/data/users/ys221/data/ABAW/origin_faces/2-30-640x360/00039.jpg'
# image = cv2.COLOR_BGR2RGB(image)
image = Image.open(filename).convert('RGB')
trans = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
image = trans(image)
# image[0, :, :] = (image[0,:,:]-0.485)/0.229
# image[1, :, :] = (image[1,:,:]-0.456)/0.224
# image[2, :, :] = (image[2,:,:]-0.406)/0.225
# image = cv2.resize(image, [64, 64], interpolation=cv2.INTER_AREA)

# image = torch.from_numpy(image).to(device=device, dtype=torch.float)
image = torch.unsqueeze(image, dim=0)
# image = image.permute(0, 3, 1, 2)
X = model(image)

neu = torch.squeeze(X)
# neu*=250.0

# img = neu.astype(np.uint8)
neu[0, :, :] = neu[0, :, :]*0.229 + 0.485
neu[1, :, :] = neu[1, :, :]*0.224 + 0.456
neu[2, :, :] = neu[2, :, :]*0.225 + 0.406
# img = img.clamp(0,1)

# r, b, g = cv2.split(img)
# img = cv2.merge([r, g, b])
# print("cv2? ", isinstance(img, np.ndarray))
# cv2.imwrite("/data/users/ys221/data/results/image.jpg", img)
vutils.save_image(neu, "/data/users/ys221/data/results/image.jpg")
