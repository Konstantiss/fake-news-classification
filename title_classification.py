import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import gc
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from time import sleep
from dataset import Titles, create_title_data_loader
from bert import TitleClassifier
from tqdm import tqdm

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_TITLE_LENGTH = 100
NUM_EPOCHS = 10
NUM_CLASSES = 2
BATCH_SIZE = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

train = pd.read_csv('./Fakeddit/train_reduced.csv')
train = train.dropna(subset=['clean_title'])

validate = pd.read_csv('./Fakeddit/validate_reduced.csv')
validate = validate.dropna(subset=['clean_title'])

train_loader = create_title_data_loader(train, tokenizer, MAX_TITLE_LENGTH, BATCH_SIZE)
validate_loader = create_title_data_loader(validate, tokenizer, MAX_TITLE_LENGTH, BATCH_SIZE)

model = TitleClassifier(NUM_CLASSES).to(device)
model.load_state_dict(torch.load('best_bert.bin'))

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

output_tensors_train = torch.tensor([0, 0])
output_tensors_validate = torch.tensor([0, 0])

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    with tqdm(data_loader, unit="batch", total=len(data_loader)) as tepoch:
        for d in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            global output_tensors_train
            #if epoch == NUM_EPOCHS and torch.count_nonzero(output_tensors_train) == 0:
            if torch.count_nonzero(output_tensors_train) == 0:
                output_tensors_train = outputs.cpu()
            #elif epoch == NUM_EPOCHS:
            else:
                output_tensors_train = torch.vstack((output_tensors_train, outputs.cpu()))

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tepoch.set_postfix(loss=loss.item(), accuracy=(correct_predictions.__float__() / n_examples))
            sleep(0.1)
            torch.cuda.empty_cache()
            gc.collect()

        return correct_predictions.__float__() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        with tqdm(data_loader, unit="batch", total=len(data_loader)) as tepoch:
            for d in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                global output_tensors_validate
                if torch.count_nonzero(output_tensors_validate) == 0:
                    output_tensors_validate = outputs.cpu()
                else:
                    output_tensors_validate = torch.vstack((output_tensors_validate, outputs.cpu()))

                _, preds = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, labels)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item(), accuracy=(correct_predictions.__float__() / n_examples))
                sleep(0.1)
                torch.cuda.empty_cache()
                gc.collect()

    return correct_predictions.__float__() / n_examples, np.mean(losses)


history = defaultdict(list)
best_accuracy = 0

for epoch in range(NUM_EPOCHS):

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        validate_loader,
        loss_fn,
        device,
        len(validate)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)


torch.save(model.state_dict(), 'bert-save.bin')
torch.save(output_tensors_train, 'bert-tensors-train.pt')
torch.save(output_tensors_validate, 'bert-tensors-validate.pt')

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history (accuracy)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])

plt.show()

plt.plot(history['train_loss'], label='train loss')
plt.plot(history['val_loss'], label='validation loss')

plt.title('Training history (loss)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])

plt.show()
