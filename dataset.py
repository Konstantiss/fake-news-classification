import pandas as pd
import os
import torch
from PIL import Image, ImageFile
import torchvision.transforms.functional_tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Images(Dataset):
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


class Titles(Dataset):

    def __init__(self, titles, labels, tokenizer, max_len):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, item):
        title = str(self.titles[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'image_title': title,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_title_data_loader(df, tokenizer, max_len, batch_size):
    ds = Titles(
        titles=df['clean_title'].to_numpy(),
        labels=df['2_way_label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )
