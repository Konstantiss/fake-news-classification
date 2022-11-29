import pandas as pd
import os
import torch
from PIL import Image, ImageFile
import torchvision.transforms.functional_tensor
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Fakeddit(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=['2_way_label'])
        self.img_ids = pd.read_csv(annotations_file, usecols=['id'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_ids.iloc[idx, 0]) + ".jpg"
        image = Image.open(img_path)
        toTensor = transforms.ToTensor()
        image = toTensor(image)
        image = torchvision.transforms.functional_tensor.convert_image_dtype(image, torch.float32)
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label