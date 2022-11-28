import numpy as np
import pandas as pd
import os
import gc
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import torchvision.transforms.functional_tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.io import read_image
from tqdm import trange
from tqdm import tqdm
from time import sleep

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


num_classes = 2
num_epochs = 1
batch_size = 28
learning_rate = 0.01

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] < 3 else x),  # convert grayscale to RGB
    transforms.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),  # convert 4 channels to 3
    transforms.RandomHorizontalFlip(p=0.5),
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((224, 224), max_size=224)
])

train_dir = './Fakeddit/train_reduced/'
label_file_train = './Fakeddit/train_reduced_more.csv'
train_data = Fakeddit(annotations_file=label_file_train, img_dir=train_dir,
                      transform=train_transforms)

valid_dir = './Fakeddit/validate/'
label_file_valid = './Fakeddit/valid_reduced.csv'
valid_data = Fakeddit(annotations_file=label_file_valid, img_dir=valid_dir,
                      transform=train_transforms)

test_dir = './Fakeddit/test_reduced/'
label_file_test = './Fakeddit/test_reduced_more.csv'
test_data = Fakeddit(annotations_file=label_file_test, img_dir=test_dir,
                     transform=test_transforms)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=batch_size, shuffle=True)

model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

# Train the model
total_step = len(train_loader)
accuracies_train = []
accuracies_validate = []
losses_train = []
losses_validate = []

start_time = time.time()

for epoch in range(num_epochs):
    with tqdm(enumerate(train_loader), unit="batch") as tepoch:
        for i, (images, labels) in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / batch_size
            accuracies_train.append(accuracy)
            loss = criterion(outputs, labels)
            losses_train.append(loss.item())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            sleep(0.1)
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

    # Validation
    with torch.no_grad():
        total_correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses_validate.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            accuracy = correct / batch_size
            accuracies_validate.append(accuracy)
            del images, labels, outputs

        total_accuracy = 100 * total_correct / total
        print('Accuracy of the network on the validation images: {} %'.format(100 * total_accuracy))

print('Total execution time: {:.4f} minutes'
      .format((time.time() - start_time) / 60))

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(losses_validate,label="val")
plt.plot(losses_train,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Training and Validation accuracy")
plt.plot(accuracies_validate,label="val")
plt.plot(accuracies_train,label="train")
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.legend()
plt.show()
