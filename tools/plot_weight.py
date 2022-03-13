import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
list_csv = glob.glob('D:/Self Improvement/Study/MRes IC/!Personal project/ys221/software/data/labels_save/expression/Validation_Set_v2/*')
# list_csv = glob.glob('/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_v2/' + '*')
# self.list_csv = [i for i in self.list_csv if len(pd.read_csv(i)) != 0]
df = pd.DataFrame()
for i in list_csv:
    df = pd.concat((df, pd.read_csv(i)), axis=0).reset_index(drop=True)
# df = creat_balance_data_ex()
list_labels_ex = np.array(df['labels_ex'].values)

counts = []
for i in range(8):
    labels = list(list_labels_ex)
    counts.append(labels.count(i))
N = len(list_labels_ex)
r = [N / counts[i] if counts[i] != 0 else 0 for i in range(8)]
s = sum(r)
weight = [r[i] / s for i in range(8)]
name_list = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise','other']
color_list = []
for i in range(len(r)):
    color_list.append(plt.cm.Set1(i))
plt.figure(figsize=(12, 8))
plt.bar(range(len(r)), r, tick_label=name_list, color=color_list)
plt.show()

plt.figure(figsize=(12, 8))
plt.bar(range(len(counts)), counts, tick_label=name_list, color=color_list)
plt.show()