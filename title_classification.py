from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

RANDOM_SEED = 42
MAX_LEN = 32
BATCH_SIZE = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = './all_train.tsv'
# since it's a tsv file the delimiter between columns is \t
df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')

labels = ['clean_title', '2_way_label']
df = df[labels]


def delete_empty_rows(dataset):
    dataset = dataset[dataset.clean_title.notnull()]
    return dataset


df = delete_empty_rows(df)
df = df[:100]

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


class CreateDataset(Dataset):

    def __init__(self, review, target, tokenizer, max_len):
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        encoding = tokenizer.encode_plus(review,
                                         add_special_tokens=True,
                                         max_length=self.max_len,
                                         return_token_type_ids=False,
                                         padding='max_length',
                                         return_attention_mask=True,
                                         return_tensors='pt', )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.target[item], dtype=torch.long)
        }


df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CreateDataset(review=df['clean_title'].to_numpy(), target=df['2_way_label'].to_numpy(), tokenizer=tokenizer,
                       max_len=max_len)

    return data.DataLoader(ds, batch_size=batch_size)


train_data_loader = create_data_loader(df_train, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


class TextClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.drop(pooled_output)
        return self.out(output)


classes = 2
model = TextClassifier(classes)
model = model.to(device)

EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_function = nn.CrossEntropyLoss().to(device)

history = defaultdict(list)
best_accuracy = 0


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


for epoch in range(EPOCHS):
    print(f'Epoch  {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, loss_function, optimizer, device, scheduler,
                                        len(df_train))

    print(f'Train loss {train_loss} accuracy {train_acc}')
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
