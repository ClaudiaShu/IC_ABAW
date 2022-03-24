import warnings
import pickle
import os
import numpy as np
import argparse
import torch
from sklearn.metrics import classification_report

from models_final import *
from data_loader import *

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import glob
import pandas as pd
from tqdm import tqdm
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--expr_txt', type=str, default='/data/users/ys221/data/ABAW/test/test.txt')
parser.add_argument('--data_dir', type=str, default='/data/users/ys221/data/ABAW/origin_faces/')
parser.add_argument('--video_dir', type=str, default='/data/users/ys221/data/ABAW/origin_videos/')

parser.add_argument('--batch_size', default=32, type=int, help='batch size')  # 256#
parser.add_argument('--num_classes', default=8, type=int, help='number classes')

parser.add_argument('--dataset', type=str, default="v2")  # ["v1","v2","v3"]
parser.add_argument('--net', type=str, default="RES")  # ["RES","INC","HRN",'C','M','T']
parser.add_argument('--mode', type=str, default="train")  # ["train","trainMixup","trainRemix"]
parser.add_argument('--file', type=str, default=f"{parser.parse_args().net}_ex_best")

args = parser.parse_args()


def create_file(list_txt):
    Expr_list = ['Neutral ', 'Anger ', 'Disgust ', 'Fear ', 'Happiness ', 'Sadness ', 'Surprise ', 'Other ']
    for idx, file in enumerate(list_txt):
        output_file_path = f'./results/test/{file}.txt'
        f_write = open(output_file_path, 'w')
        f_write.writelines(Expr_list)
        f_write.writelines(str('\n'))
        f_write.close()
    return 0


def get_file(id, list_txt):
    for idx, file in enumerate(list_txt):
        if id == idx:
            return file


def get_model(file_name, model_name):
    checkpoint = torch.load(file_name)
    net = checkpoint['net']

    if model_name == "RES":
        model = BaselineRES(num_classes=args.num_classes)
    elif model_name == "INC":
        model = BaselineINC(num_classes=args.num_classes)
    elif model_name == "HRN":
        model = BaselineHRN(num_classes=args.num_classes)
    elif model_name == "RES_C":
        model = BaselineRES_C(num_classes=args.num_classes)
    elif model_name == "INC_C":
        model = BaselineINC_C(num_classes=args.num_classes)
    elif model_name == "RES_M":
        model = BaselineRES_M(num_classes=args.num_classes)
    elif model_name== "INC_M":
        model = BaselineINC_M(num_classes=args.num_classes)
    elif model_name == "RES_T":  # tri
        model = BaselineRES_T(num_classes=args.num_classes)
    elif model_name == "INC_T":  # tri
        model = BaselineINC_T(num_classes=args.num_classes)
    else:
        model = Baseline(num_classes=args.num_classes)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(net.module.state_dict())
    model.to(device)
    return model


def read_txt(txt_file):
    with open(txt_file, 'r') as f:
        videos = f.readlines()
    videos = [x.strip() for x in videos]
    return videos


def refine_frames_paths(frames, cap, save=False):
    # Get the cropped and aligned ids
    frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in frames]
    # Get the number of original frames
    length = int(cap.get(7)) + 1  # number of total frames
    # Save dir
    pre_dir = '/'.join(frames[0].split('/')[:-3])
    save_dir = os.path.join(pre_dir, 'origin_frames/')
    extra_frame_ids = []

    if (len(frames) > length) and np.abs(len(frames) - length) == 1:
        length = len(frames)

    if len(frames) == length:
        return frames, extra_frame_ids
    elif len(frames) < length:

        prefix = '/'.join(frames[0].split('/')[:-1])
        print(f"Updating the {prefix} video frames")
        video = prefix.split('/')[-1]
        for i in range(length):
            if i not in frames_ids:
                extra_frame_ids.append(i)
                # Get the missing frame
                if save:
                    cap.set(1, i)
                    ret, frame = cap.read()
                    os.makedirs(save_dir+video+"/", exist_ok=True)
                    cv2.imwrite(save_dir+video+"/{0:05d}.jpg".format(i + 1), frame)
        frames_ids.extend(extra_frame_ids)
        frames_ids = sorted(frames_ids)

        return_frames = [prefix + '/{0:05d}.jpg'.format(id + 1) for id in frames_ids]
        return return_frames, extra_frame_ids
    else:
        raise ValueError("Number of frames larger than the cv2 video length")


def prediction_by_pic():
    checkpoint = torch.load(
        f'./checkpoint/expression/{args.net}_{args.dataset}_{args.mode}/{args.file}_0.pth')
    net = checkpoint['net']
    #
    model = BaselineRES(num_classes=args.num_classes)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(net.state_dict())
    model.to(device)

    ####################################################
    ########### data processing starts here ############
    ####################################################
    data_file = {}
    mode = 'Test_Set'
    txt_file = args.expr_txt
    videos = read_txt(txt_file)
    for idx, video in enumerate(videos):
        name = video
        frames_paths = sorted(glob.glob(os.path.join(args.data_dir, name, '*.jpg')))
        if '_left' in name:
            video_name = name[:-5]
        elif '_right' in name:
            video_name = name[:-6]
        else:
            video_name = name
        # Go through the original video
        video_path = glob.glob(os.path.join(args.video_dir, 'batch*', video_name + ".*"))[0]
        cap = cv2.VideoCapture(video_path)
        frames_paths, extra_frame_ids = refine_frames_paths(frames_paths, cap)
        length = len(frames_paths)

        Expr_list = ['Neutral ', 'Anger ', 'Disgust ', 'Fear ', 'Happiness ', 'Sadness ', 'Surprise ', 'Other ']
        data_dict = {'label': np.zeros(length), 'path': frames_paths, 'frames_ids': np.arange(length)}

        # fps
        video = cv2.VideoCapture(video_path)
        fps = int(np.round(video.get(cv2.CAP_PROP_FPS)))
        data_dict.update({'fps': [fps] * len(frames_paths)})
        data_dict.update({'video': [name] * len(frames_paths)})
        data_file[name] = data_dict

        # prediction
        output_file_path = f'./results/test/{name}.txt'
        f_write = open(output_file_path, 'w+')
        f_write.writelines(Expr_list)
        f_write.writelines(str('\n'))
        model.eval()
        for frame_path in tqdm(frames_paths, total=len(frames_paths), desc=f'Process prediction on video {idx}'):
            # Iterate through selected frames in each video
            if os.path.exists(frame_path):
                image = Image.open(frame_path).convert('RGB')
                img = test_transform(image)
                img = img.unsqueeze(dim=0).to(device)
                pred = model(img)
                pred = torch.argmax(F.softmax(pred), dim=1)
                pred_label = pred.detach().cpu().numpy()

                f_write.writelines(str(pred_label[0]) + '\n')
            else:
                f_write.writelines(str('nan'))
        f_write.close()

    save_path = os.path.join('.', 'test_set.pkl')
    pickle.dump(data_file, open(save_path, 'wb'))


def test(valid_loader, model, file):
    model.eval()
    with torch.no_grad():
        # test_videos = torch.Tensor().cuda()
        # test_images = torch.Tensor().cuda()
        test_preds = torch.Tensor().cuda()

        for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Valid_mode {file}'):
            images = samples['images'].to(device).float()

            pred_cat = model(images)

            pred_cat = F.softmax(pred_cat)

            pred_cat = torch.argmax(pred_cat, dim=1)
            pred = pred_cat.detach()

            test_preds = torch.cat([test_preds, pred], dim=0)

        return test_preds


def test_tri(valid_loader, model, file):
    model.eval()
    with torch.no_grad():
        # test_videos = torch.Tensor().cuda()
        # test_images = torch.Tensor().cuda()
        test_preds = torch.Tensor().cuda()

        for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Valid_mode {file}'):
            images = samples['images'].to(device).float()
            # image_id = samples['image_id'].to(device)
            # video_id = samples['video_id'].to(device)

            pred1, pred2, pred3, _ = model(images)

            pred_cat = F.softmax(pred_cat)
            pred_cat = torch.argmax(pred_cat, dim=1)
            pred = pred_cat.detach()

            # test_videos = torch.cat([test_videos, video_id], dim=0)
            # test_images = torch.cat([test_images, image_id], dim=0)
            test_preds = torch.cat([test_preds, pred], dim=0)

        return test_preds


def get_extra_id():
    # txt_file = args.expr_txt
    txt_file = '/data/users/ys221/data/ABAW/test/test_debug.txt'
    list_txt = read_txt(txt_file)
    sample = {}
    for idx, video in enumerate(list_txt):
        name = video
        frames_paths = sorted(glob.glob(os.path.join(args.data_dir, name, '*.jpg')))
        if '_left' in name:
            video_name = name[:-5]
        elif '_right' in name:
            video_name = name[:-6]
        else:
            video_name = name
        # Go through the original video
        video_path = glob.glob(os.path.join(args.video_dir, 'batch*', video_name + ".*"))[0]
        cap = cv2.VideoCapture(video_path)
        frames_paths, extra_frame_ids = refine_frames_paths(frames_paths, cap)
        sample[name] = {}
        sample[name]['num'] = len(frames_paths)
        sample[name]['extra_id'] = extra_frame_ids
    return sample


def prediction_by_batch():

    net = 'RES_M'
    file = f'./checkpoint/expression/{net}_v2_{args.mode}/{net}_ex_best_1.pth'

    model = get_model(file, net)

    # seed_everything()
    # Write txt
    txt_file = args.expr_txt
    # txt_file = '/data/users/ys221/data/ABAW/test/test_debug.txt'
    list_txt = read_txt(txt_file)
    sample = get_extra_id()
    for file in list_txt:

        test_dataset = Aff2_Dataset_test(csv=f'/data/users/ys221/data/ABAW/test/Test_Set/{file}.csv',
                                         transform=test_transform)

        valid_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=0,
                                  shuffle=False,
                                  drop_last=False)

        test_preds = test(valid_loader, model, file)

        test_preds = test_preds.detach().cpu().numpy().astype(int)

        len_video = sample[file]['num']
        extra_id = sample[file]['extra_id']
        output = np.zeros(len_video, dtype=int)
        pred_id = 0
        for id in range(len_video):
            if id not in extra_id:
                output[id] = test_preds[pred_id]
                pred_id += 1


        df = pd.DataFrame(output, columns=['Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other'])
        df.to_csv(f'./results/test/{file}.txt', index=False)


if __name__ == '__main__':
    # prediction_by_pic()
    prediction_by_batch()

