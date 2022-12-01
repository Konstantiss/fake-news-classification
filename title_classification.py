import pandas as pd
import torch.utils.data
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

data_path = './all_train.tsv'
# since it's a tsv file the delimiter between columns is \t
df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')

labels = ['clean_title', '2_way_label', 'image_url']

df = df[labels]


def delete_empty_rows(dataset):
    dataset = dataset[dataset.image_url.notnull()]
    dataset = dataset[dataset.clean_title.notnull()]
    return dataset


df = delete_empty_rows(df)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [labels[label] for label in df['2_way_label']]
        self.texts = [tokenizer(text, padding='max_length', max_length=100, truncation=True, return_tensors="pt") for
                      text in df['clean_title']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_text(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_text(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_texts, batch_labels

np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)),int(.9*len(df))])
print(len(df_train),len(df_val), len(df_test))

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU

    def forward(self, input_id, mask):
        _, pooled_ouput = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_ouput = self.dropout(pooled_ouput)
        linear_ouput = self.linear(dropout_ouput)
        final_layer = self.relu(linear_ouput)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id,mask)
            batch_loss = criterion(output,train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                            | Val Loss: {total_loss_val / len(val_data): .3f} \
                            | Val Accuracy: {total_acc_val / len(val_data): .3f}')

dataset = Dataset(df)


EPOCHS = 2
model = BertClassifier()
learning_rate = 1e-6

train(model, df_train, df_val,learning_rate,epochs=EPOCHS)