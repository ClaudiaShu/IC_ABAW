import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch.nn
from sklearn.manifold import TSNE
import matplotlib as mpl
import torch
import glob
import cv2
import os

from torchvision import transforms

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

trans = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

'''
data loader
'''
list_csv = glob.glob("/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_v2/" + '*')
df = pd.DataFrame()
for i in list_csv:
    df = pd.concat((df, pd.read_csv(i)), axis=0).reset_index(drop=True)
list_labels_ex = np.array(df['labels_ex'].values)
list_image_id = np.array(df['image_id'].values)

# images = []
images = torch.Tensor()
labels = []
idn = []
# L1 = random.sample(range(1, 421093), 1000)
for i in range(1000):
    j = i*200
    image = Image.open(list_image_id[j]).convert('RGB')
    image = trans(image)
    image = torch.unsqueeze(image, dim=0)
    images = torch.cat([images, image], dim=0)
    labels.append(int(list_labels_ex[j]))
    seq = list_image_id[j].split('/')[7]
    idn.append(int(seq.split('-')[0]))
# labels = np.stack(labels)
# idn = np.stack(idn)

'''
models
'''
model = BaseLineCycletest()

# model = BaseLineR3D(num_classes=8)
# model = BaseLineCycle(num_classes=8)
# model = Baseline(num_classes=8)
# model = BaselineALEX(num_classes=8)

# checkpoint = torch.load('/data/users/ys221/software/IC_ABAW2022/checkpoint/ex/HRNL/v2_CE_SGD/CELS64_ex_best.pth',
#                         map_location='cpu')
# model = checkpoint['net']
# model.fc2 = torch.nn.Identity()

model.to(device)
# '''
X = model(images).cpu().detach().numpy()
print(X.shape)

# tsne = TSNE(n_components=2, perplexity=50, verbose=2)
tsne = TSNE(n_components=2, perplexity=100, learning_rate=1000, n_iter=1000, random_state=0)
X_tsne = tsne.fit_transform(X)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化


################### plot ID & emo ##################
# plt.figure(figsize=(20, 20))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(idn[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 8})
# plt.xticks([])
# plt.yticks([])
# plt.show()

################### plot ID ##################
plt.figure(figsize=(40, 20))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=idn[:], cmap='plasma')
plt.xticks([])
plt.yticks([])
################### plot emo ##################
plt.subplot(122)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels[:], cmap='rainbow')
plt.xticks([])
plt.yticks([])
plt.show()



