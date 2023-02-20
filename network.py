import gc
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from time import sleep
from resnet import ResNet50
from resnet import ResNet18
from dataset import Images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
num_epochs = 20
batch_size = 24
learning_rate = 0.001

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] < 3 else x),  # convert images to 3 channels
    transforms.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),  # convert images to 3 channels
    transforms.RandomHorizontalFlip(p=0.5),
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((224, 224), max_size=224)
])

train_dir = './Fakeddit/train_reduced/'
label_file_train = './Fakeddit/train_reduced.csv'
train_data = Images(annotations_file=label_file_train, img_dir=train_dir,
                    transform=train_transforms)

valid_dir = './Fakeddit/validate_reduced/'
label_file_valid = './Fakeddit/validate_reduced.csv'
valid_data = Images(annotations_file=label_file_valid, img_dir=valid_dir,
                    transform=train_transforms)

test_dir = './Fakeddit/test_reduced/'
label_file_test = './Fakeddit/test_reduced.csv'
test_data = Images(annotations_file=label_file_test, img_dir=test_dir,
                   transform=test_transforms)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=batch_size, shuffle=True)

model = ResNet50(num_classes=2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)

# Train the model
total_step = len(train_loader)
accuracies_train_epoch = []
accuracies_validate_epoch = []
losses_train_epoch = []
losses_validate_epoch = []

start_time = time.time()

output_tensors_train = torch.tensor([0, 0])
output_tensors_validate = torch.tensor([0, 0])

for epoch in range(num_epochs):
    accuracies_train = []
    accuracies_validate = []
    losses_train = []
    losses_validate = []
    print("Learning rate: ", optimizer.param_groups[0]['lr'])
    model.train()
    with tqdm(enumerate(train_loader), unit="batch", total=len(train_loader)) as tepoch:
        for i, (images, labels) in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)

            #if epoch == num_epochs and i == 0:
            if i == 0:
                output_tensors_train = outputs.cpu()
            #elif epoch == num_epochs:
            else:
                output_tensors_train = torch.vstack((output_tensors_train, outputs.cpu()))

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
    accuracies_train_epoch.append(sum(accuracies_train) / len(accuracies_train))
    losses_train_epoch.append(sum(losses_train) / len(losses_train))

    # Validation
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total = 0
        with tqdm(enumerate(valid_loader), unit="batch", total=len(valid_loader)) as tepoch:
            for i, (images, labels) in tepoch:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                if i == 0:
                    output_tensors_validate = outputs.cpu()
                else:
                    output_tensors_validate = torch.vstack((output_tensors_validate, outputs.cpu()))

                loss = criterion(outputs, labels)
                losses_validate.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct = (predicted == labels).sum().item()
                total_correct += correct
                accuracy = correct / batch_size
                accuracies_validate.append(accuracy)
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                sleep(0.1)
                del images, labels, outputs

            accuracies_validate_epoch.append(sum(accuracies_validate) / len(accuracies_validate))
            losses_validate_epoch.append(sum(losses_validate) / len(losses_validate))
            scheduler.step(sum(losses_validate) / len(losses_validate))

torch.save(output_tensors_train, 'resnet-tensors-train.pt')
torch.save(output_tensors_validate, 'resnet-tensors-validate.pt')
torch.save(model.state_dict(), 'resnet-save.bin')
print('Total execution time: {:.4f} minutes'
      .format((time.time() - start_time) / 60))

plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(losses_validate_epoch, label="val")
plt.plot(losses_train_epoch, label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Training and Validation accuracy")
plt.plot(accuracies_validate_epoch, label="val")
plt.plot(accuracies_train_epoch, label="train")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
