import os
import pandas as pd

def dropUnusedRows(annotations_file, img_dir):
    dataset = pd.read_csv(annotations_file, sep='\t')
    indexes_to_drop = []
    for index, row in dataset.iterrows():
        file_list = os.listdir(img_dir)
        file_name = row["id"] + ".jpg"
        if file_name not in file_list:
            indexes_to_drop.append(index)
    dataset.drop(dataset.index[indexes_to_drop], inplace=True)
    return dataset

train_dir = './Fakeddit/train/'
label_file_train = './Fakeddit/all_train.tsv'
train_data = dropUnusedRows(label_file_train, train_dir)
train_data.to_csv('./Fakeddit/train_reduced.csv')

test_dir = './Fakeddit/test/'
label_file_test = './Fakeddit/all_test_public.tsv'
test_data = dropUnusedRows(label_file_train, train_dir)
test_data.to_csv('./Fakeddit/test_reduced.csv')
