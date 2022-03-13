import ast
import os.path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils import *
import pdb

# data
transform1 = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()])
transform2 = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor()])
transform3 = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(30),
    transforms.GaussianBlur(3),
    # transforms.RandomAutocontrast(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor()])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.GaussianBlur(3), # add
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_transform_CYC = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.GaussianBlur(3), # add
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
test_transform_CYC = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class Aff2_Dataset_static_shuffle(Dataset):
    def __init__(self, root, transform, type_partition, df=None):
        super(Aff2_Dataset_static_shuffle, self).__init__()
        if root:
            self.list_csv = glob.glob(root + '*')
            # self.list_csv = [i for i in self.list_csv if len(pd.read_csv(i)) != 0]
            self.df = pd.DataFrame()

            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.df = pd.concat((self.df, pd.read_csv(i)), axis=0).reset_index(drop=True)
        else:
            self.df = df
        self.transform = transform
        self.type_partition = type_partition
        if self.type_partition == 'ex':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
        elif self.type_partition == 'au':
            self.list_labels_au = np.array(self.df['labels_au'].values)
        else:
            self.list_labels_va = np.array(self.df['labels_va'].values)

        self.list_image_id = np.array(self.df['image_id'].values)

    def __getitem__(self, index):
        if self.type_partition == 'ex':
            label = self.list_labels_ex[index]
        elif self.type_partition == 'au':
            label = ast.literal_eval(self.list_labels_au[index])
        else:
            label = ast.literal_eval(self.list_labels_va[index])
            # label = (label+1)/2.0
        image = cv2.imread(self.list_image_id[index])[..., ::-1]
        sample = {
            'images': self.transform(image),
            'labels': torch.tensor(label)
        }
        return sample

    def __len__(self):
        return len(self.df)


class Aff2_Dataset_series_shuffle(Dataset):
    def __init__(self, root, transform, type_partition, length_seq, df=None):
        super(Aff2_Dataset_series_shuffle, self).__init__()
        if root:
            self.list_csv = glob.glob(root + '*')
            # import pdb; pdb.set_trace()
            self.df = pd.DataFrame()
            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.one_df = pd.read_csv(i)
                self.one_df = pad_if_need(self.one_df, length_seq)
                self.df = pd.concat((self.df, self.one_df), axis=0).reset_index(drop=True)
        else:
            self.df = df
        self.transform = transform
        self.type_partition = type_partition
        if self.type_partition == 'ex':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
        elif self.type_partition == 'au':
            self.list_labels_au = np.array(self.df['labels_au'].values)
        else:
            self.list_labels_va = np.array(self.df['labels_va'].values)

        self.list_image_id = np.array(self.df['image_id'].values)
        self.length = length_seq

    def __getitem__(self, index):
        images = []
        labels = []
        for i in range(index * self.length, index * self.length + self.length):
            if self.type_partition == 'ex':
                label = self.list_labels_ex[i]
            elif self.type_partition == 'au':
                label = ast.literal_eval(self.list_labels_au[index])
            else:
                label = ast.literal_eval(self.list_labels_va[index])
                # label = (label+1)/2.0

            image = cv2.imread(self.list_image_id[i])[..., ::-1]
            images.append(self.transform(image))
            labels.append(label)
        # import pdb; pdb.set_trace()
        images = np.stack(images)
        labels = int(np.array(labels).mean())

        # assert labels.is_integer()
        # print(images.shape)
        # print(labels.shape)
        sample = {
            'images': torch.tensor(images),
            'labels': torch.tensor(labels)
        }
        return sample

    def __len__(self):
        return len(self.df) // self.length


class Aff2_Dataset_static_facemask(Dataset):
    def __init__(self, root, transform, type_partition, df=None):
        super(Aff2_Dataset_static_facemask, self).__init__()
        if root:
            self.list_csv = glob.glob(root + '*')
            # self.list_csv = [i for i in self.list_csv if len(pd.read_csv(i)) != 0]
            self.df = pd.DataFrame()

            for i in tqdm(self.list_csv, total=len(self.list_csv)):
                self.df = pd.concat((self.df, pd.read_csv(i)), axis=0).reset_index(drop=True)
        else:
            self.df = df
        self.transform = transform
        self.type_partition = type_partition
        if self.type_partition == 'ex':
            self.list_labels_ex = np.array(self.df['labels_ex'].values)
        elif self.type_partition == 'au':
            self.list_labels_au = np.array(self.df['labels_au'].values)
        else:
            self.list_labels_va = np.array(self.df['labels_va'].values)

        self.list_image_id = np.array(self.df['image_id'].values)

    def __getitem__(self, index):
        if self.type_partition == 'ex':
            label = self.list_labels_ex[index]
        elif self.type_partition == 'au':
            label = ast.literal_eval(self.list_labels_au[index])
        else:
            label = ast.literal_eval(self.list_labels_va[index])
            # label = (label+1)/2.0

        image = cv2.imread(self.list_image_id[index])[..., ::-1]
        dir = '/data/users/ys221/data/ABAW/facemasks'
        video = self.list_image_id[index].split('/')[6]
        frame = self.list_image_id[index].split('/')[7]
        facemask = cv2.imread(os.path.join(dir, video, frame), cv2.IMREAD_GRAYSCALE)[..., ::-1]

        sample = {
            'images': self.transform(image),
            'masks': torch.tensor(facemask),
            'labels': torch.tensor(label)
        }
        return sample

    def __len__(self):
        return len(self.df)


class DFEW_Dataset(Dataset):
    def __init__(self, args, phase):
        # Basic info
        self.args = args
        self.phase = phase

        # File path
        label_df_path = os.path.join(self.args.data_root,
                                     "label/{data_type}_{phase}set_{fold_idx}.csv".format(data_type=self.args.data_type,
                                                                                          phase=self.phase,
                                                                                          fold_idx=str(
                                                                                              int(self.args.fold_idx))))
        label_df = pd.read_csv(label_df_path)

        # Imgs & Labels
        self.names = label_df['video_name']
        self.videos_path = [os.path.join(self.args.data_root, "data/{name}".format(name=str(ele).zfill(5)))
                            for ele in self.names]
        self.single_labels = torch.from_numpy(np.array(label_df['label']))

        # Transforms
        self.my_transforms_fun_dataAugment = self.my_transforms_fun_dataAugment()
        self.my_transforms_te = self.my_transforms_fun_te()

    def __len__(self):
        return len(self.single_labels)

    def __getitem__(self, index):
        imgs_per_video = glob.glob(self.videos_path[index] + '/*')
        imgs_per_video = sorted(imgs_per_video)
        imgs_idx = self.generate_index(nframe=self.args.nframe,
                                       idx_start=0,
                                       idx_end=len(imgs_per_video) - 1,
                                       phase=self.phase,
                                       isconsecutive=self.args.isconsecutive)
        data = torch.zeros(3, self.args.nframe, self.args.size_Resize_te, self.args.size_Resize_te)
        for i in range(self.args.nframe):
            img = Image.open(imgs_per_video[imgs_idx[i]])
            if self.phase == "train":
                if self.args.train_data_augment == True:
                    img = self.my_transforms_fun_dataAugment(img)
                else:
                    img = self.my_transforms_te(img)
            if self.phase == "test":
                img = self.my_transforms_te(img)
            data[:, i, :, :] = img

        single_label = self.single_labels[index]

        return data, single_label

    def generate_index(self, nframe, idx_start, idx_end, phase, isconsecutive):
        if (idx_end - idx_start + 1) < nframe:
            idx_list_tmp = list(range(idx_start, idx_end + 1))
            idx_list = []
            for j in range(100):
                idx_list = idx_list + idx_list_tmp
                if len(idx_list) >= nframe:
                    break
            if isconsecutive == True:
                if self.phase == "train":
                    idx_s = random.randint(idx_start, idx_end - nframe)
                else:
                    idx_s = int(idx_end - nframe - idx_start)
                idx_tmp = list(range(idx_s, idx_s + nframe))
                idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]
            if isconsecutive == False:
                if self.phase == "train":
                    idx_tmp = random.sample(range(len(idx_list)), nframe)
                    idx_tmp.sort()
                    idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]
                else:
                    idx_tmp = np.linspace(0, len(idx_list) - 1, nframe).astype(int)
                    idx = [idx_list[idx_tmp[jj]] for jj in range(len(idx_tmp))]

        if (idx_end - idx_start + 1) >= nframe:
            if isconsecutive == True:
                if self.phase == "train":
                    idx_s = random.randint(idx_start, idx_end - nframe)
                else:
                    idx_s = int(idx_end - nframe - idx_start)
                idx = list(range(idx_s, idx_s + nframe))
            if isconsecutive == False:
                if self.phase == "train":
                    idx = random.sample(range(idx_start, idx_end + 1), nframe)
                    idx.sort()
                else:
                    idx = np.linspace(idx_start, idx_end, nframe).astype(int)

        return idx

    def my_transforms_fun_dataAugment(self):
        my_img_transforms_list = []
        if self.args.Flag_RandomRotation:        my_img_transforms_list.append(
            transforms.RandomRotation(degrees=self.args.degree_RandomRotation))
        if self.args.Flag_CenterCrop:            my_img_transforms_list.append(
            transforms.CenterCrop(self.args.size_CenterCrop))
        if self.args.Flag_RandomResizedCrop:     my_img_transforms_list.append(
            transforms.RandomResizedCrop(self.args.size_RandomResizedCrop))
        if self.args.Flag_RandomHorizontalFlip:  my_img_transforms_list.append(
            transforms.RandomHorizontalFlip(p=self.args.prob_RandomHorizontalFlip))
        if self.args.Flag_RandomVerticalFlip:    my_img_transforms_list.append(
            transforms.RandomVerticalFlip(p=self.args.prob_RandomVerticalFlip))
        my_img_transforms_list.append(transforms.ToTensor())

        my_tensor_transforms_list = []
        if (self.args.model_pretrain == True) and (
                self.args.pretrained_weights == "ImageNet"): my_tensor_transforms_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        if self.args.Flag_RamdomErasing:        my_tensor_transforms_list.append(
            transforms.RandomErasing(p=self.args.prob_RandomErasing))

        my_transforms_list = my_img_transforms_list + my_tensor_transforms_list
        my_transforms = transforms.Compose(my_transforms_list)

        return my_transforms

    def my_transforms_fun_te(self):
        my_img_transforms_list = []
        if self.args.Flag_Resize_te == True: my_img_transforms_list.append(transforms.Resize(self.args.size_Resize_te))
        my_img_transforms_list.append(transforms.ToTensor())

        my_tensor_transforms_list = []
        if (self.args.model_pretrain == True) and (
                self.args.pretrained_weights == "ImageNet"): my_tensor_transforms_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        my_transforms_list = my_img_transforms_list + my_tensor_transforms_list
        my_transforms = transforms.Compose(my_transforms_list)

        return