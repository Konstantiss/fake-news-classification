import torch
import torch.nn as nn
import pandas as pd
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

optimal_layer_size = 224
k_way = 2


class SoftmaxClassifier(nn.Module):
    def __init__(self, k_way=2):
        super(SoftmaxClassifier, self).__init__()
        self.input_layer = nn.Linear(k_way, optimal_layer_size)
        self.hidden_layer = nn.Linear(optimal_layer_size, optimal_layer_size)
        self.output_layer = nn.Linear(optimal_layer_size, k_way)
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.Softmax()

    def forward(self, X):
        X = self.input_layer(X)
        X = self.hidden_activation(X)
        X = self.hidden_layer(X)
        X = self.hidden_activation(X)
        X = self.output_layer(X)
        X = self.output_activation(X)
        return X


resnet_train = torch.load('resnet-tensors-train.pt')
resnet_val = torch.load('resnet-tensors-validate.pt')
bert_train = torch.load('bert-tensors-train.pt')
bert_val = torch.load('bert-tensors-validate.pt')

missing_labels_train = resnet_train.shape[0] - bert_train.shape[0]
missing_labels_val = resnet_val.shape[0] - bert_val.shape[0]

bert_train = torch.cat((bert_train, torch.zeros((missing_labels_train, k_way))))
bert_val = torch.cat((bert_val, torch.zeros((missing_labels_val, k_way))))

train_X = torch.maximum(resnet_train, bert_train)
train_labels = pd.read_csv('train_reduced.csv', usecols=['2_way_label']).to_numpy()
train_labels = torch.squeeze(torch.from_numpy(train_labels))
train_dataset = TensorDataset(train_X[:14718], train_labels)

val_X = torch.maximum(resnet_val, bert_val)
val_labels = pd.read_csv('validate_reduced.csv', usecols=['2_way_label']).to_numpy()
val_labels = torch.squeeze(torch.from_numpy(val_labels))
val_dataset = TensorDataset(val_X, val_labels)

model = SoftmaxClassifier(k_way=k_way)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

epochs = 100

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

for e in range(epochs):
    running_loss = 0
    model.train()
    for vector, labels in train_loader:
        optimizer.zero_grad()
        output = model(vector)
        loss = loss_func(output, labels)
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
    else:
        model.eval()
        val_output = model(val_X)
        output_label = torch.argmax(val_output, dim=1)
        val_loss = loss_func(val_output, val_labels)
        val_accuracy = torch.sum(output_label == val_labels)

        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))
        print(f"Eval loss = {val_loss}, accuracy = {val_accuracy/val_labels.shape[0]}")