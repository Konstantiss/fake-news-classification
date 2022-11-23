import numpy as np
import pandas as pd
import os
import gc
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.io import read_image


class Fakeddit(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=['2_way_label'], sep='\t')
        self.img_ids = pd.read_csv(annotations_file, usecols=['id'], sep='\t')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_ids.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label


train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dir = './Fakeddit/train/'
label_file_train = './Fakeddit/all_train.tsv'
train_data = Fakeddit(annotations_file=label_file_train, img_dir=train_dir,
                      transform=train_transforms)

test_dir = './Fakeddit/test/'
label_file_test = './Fakeddit/all_test_public.tsv'
test_data = Fakeddit(annotations_file=label_file_test, img_dir=train_dir,
                      transform=test_transforms)

print(train_data.__len__())
print(test_data.__len__())
