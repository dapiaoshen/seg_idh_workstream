import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms
import pickle



def cc2weight(cc, w_min: float = 1., w_max: float = 2e5):
    weight = torch.zeros_like(cc, dtype=torch.float32)
    cc_items = torch.unique(cc)
    K = len(cc_items) - 1
    N = torch.prod(torch.tensor(cc.shape))
    for i in cc_items:
        weight[cc == i] = N / ((K + 1) * torch.sum(cc == i))
    return torch.clip(weight, w_min, w_max)

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class Crop_test(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        # 获取包含肿瘤区域的中心
        tumor_region = np.where(np.isin(label, [1, 2, 4]))
        if len(tumor_region[0]) == 0:  # 如果没有肿瘤区域，返回原始样本
            return sample

        # 计算肿瘤区域的中心点
        center_h = int(np.mean(tumor_region[0]))
        center_w = int(np.mean(tumor_region[1]))
        center_d = int(np.mean(tumor_region[2]))


        H_max, W_max, D_max = image.shape[:3]

        # 确定裁剪区域的起始点
        H_start = max(0, center_h - 64)
        W_start = max(0, center_w - 64)
        D_start = max(0, center_d - 64)

        # 确保裁剪区域在图像边界内
        H_start = min(H_start, H_max - 128)
        W_start = min(W_start, W_max - 128)
        D_start = min(D_start, D_max - 128)

        # 防止起始点为负
        H_start = max(0, H_start)
        W_start = max(0, W_start)
        D_start = max(0, D_start)

        image_crop = image[H_start:H_start + 128, W_start:W_start + 128, D_start:D_start + 128, ...]
        label_crop = label[H_start:H_start + 128, W_start:W_start + 128, D_start:D_start + 128]

        pad_h = max(0, 128 - image_crop.shape[0])
        pad_w = max(0, 128 - image_crop.shape[1])
        pad_d = max(0, 128 - image_crop.shape[2])

        image_crop = np.pad(image_crop, ((0, pad_h), (0, pad_w), (0, pad_d), (0, 0)), mode='constant',
                            constant_values=0)
        label_crop = np.pad(label_crop, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant', constant_values=0)

        return {'image': image_crop, 'label': label_crop, 'idh': sample['idh'], 'radiomic': sample['radiomic']}






class guiyihua(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        mean = np.mean(image, axis=(0, 1, 2))
        std = np.std(image, axis=(0, 1, 2))

        # 对每个像素点进行归一化
        for i in range(4):
            if std[i] == 0:
                print(f"Warning: Standard deviation for channel {i} is zero. Using mean subtraction only.")
                image[:, :, :, i] = image[:, :, :, i] - mean[i]
            else:
                image[:, :, :, i] = (image[:, :, :, i] - mean[i]) / std[i]

        return {'image': image, 'label': label, 'idh': sample['idh'], 'radiomic': sample['radiomic']}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        image = torch.from_numpy(image).float()
        return {'image': image, 'label': label,'idh':sample['idh'], 'radiomic':sample['radiomic']}

def transform_valid(sample):
    trans = transforms.Compose([
        guiyihua(),
        # MaxMinNormalization(),
        Crop_test(),
        ToTensor()
    ])

    return trans(sample)




class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train'):
        self.lines = []
        paths, names, targets = [], [],[]
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        image, label, idh, radiomic = pkload(path + 'idhgrade.pkl')
        name = os.path.basename(os.path.dirname(path))
        if not np.any(label != 0):
            print("Label contains only zeros.")
            print(name)
        sample = {'image': image, 'label': label, 'idh': idh, 'radiomic': radiomic}
        sample = transform_valid(sample)
        return sample['image'], sample['label'], sample['idh'], sample['radiomic'], name


    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]