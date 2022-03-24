import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.function import *



def take_images(data_file_name):
    images = glob.glob(data_file_name + '/*.*')
    images.sort()
    number_id = [os.path.split(image)[1].split('.')[0] for image in images]
    number_id = np.array(number_id, int)
    return images, number_id


def creat_df_test(frames_paths, paths_save):
    # write csv
    df = pd.DataFrame(frames_paths, columns=['image_id'])
    df.to_csv(paths_save)


def take_name_video(dir_path):
    return os.path.split(dir_path)[1].split('.')[0]


if __name__ == '__main__':
    txt_file = '/data/users/ys221/data/ABAW/test/test.txt'
    csv_file = '/data/users/ys221/data/ABAW/test/test.csv'
    dir_images = '/data/users/ys221/data/ABAW/origin_faces/'
    dir_save_dir = '/data/users/ys221/data/ABAW/test/'
    set_data = 'Test_Set'

    path_save = os.path.join(dir_save_dir, set_data)
    os.makedirs(path_save, exist_ok=True)

    videos = read_txt(txt_file)
    for video in tqdm(videos):
        name = video
        frames_paths = sorted(glob.glob(os.path.join(dir_images, name, '*.jpg')))
        paths_save = os.path.join(path_save, name + '.csv')

        creat_df_test(frames_paths, paths_save)

