import os
import pandas as pd
import glob
from multiprocessing import Pool
from PIL import Image


def checkSingleImage(f):
    try:
        im = Image.open(f)
        im.verify()
        im.close()
        # DEBUG: print(f"OK: {f}")
        return
    except (IOError, OSError, Image.DecompressionBombError):
        os.remove(f)
        return f


def findCorruptImages(img_dir):
    # Create a pool of processes to check images
    p = Pool()

    # Create a list of images to process
    files = [f for f in glob.glob(img_dir + "/*.jpg")]

    print(f"Files to be checked: {len(files)}")

    # Map the list of files to check onto the Pool
    result = p.map(checkSingleImage, files)

    # Filter out None values representing files that are ok, leaving just corrupt ones
    result = list(filter(None, result))
    print(f"Num corrupt files: {len(result)}")


def dropUnusedRows(annotations_file, img_dir):
    dataset = pd.read_csv(annotations_file)
    file_list = os.listdir(img_dir)
    file_list = {file.replace('.jpg', '') for file in file_list}
    dataset = dataset[dataset.id.isin(file_list)]
    return dataset

def removeDatasetBias(annotations_file, img_dir):
    dataset = pd.read_csv(annotations_file)
    num_of_zeros = (dataset['2_way_label'] == 0).sum()
    num_of_ones = (dataset['2_way_label'] == 1).sum()
    if num_of_zeros > num_of_ones:
        diff = num_of_zeros - num_of_ones
        zero_ids = dataset['id'].loc[dataset['2_way_label'] == 0]
        zero_ids = zero_ids.head(diff)
        dataset = dataset[~dataset.id.isin(zero_ids)]
        zero_ids = zero_ids.astype(str) + '.jpg'
        for img_id in zero_ids:
            if os.path.exists(img_dir + img_id):
                os.remove(img_dir + img_id)
    elif num_of_zeros < num_of_ones:
        diff = num_of_ones - num_of_zeros
        one_ids = dataset['id'].loc[dataset['2_way_label'] == 1]
        one_ids = one_ids.head(diff)
        dataset = dataset[~dataset.id.isin(one_ids)]
        one_ids = one_ids.astype(str) + '.jpg'

        for img_id in one_ids:
            if os.path.exists(img_dir + img_id):
                os.remove(img_dir + img_id)

    return dataset



train_dir = './Fakeddit/train/'
label_file_train = './Fakeddit/train.csv'
train_dir_reduced = './Fakeddit/train_reduced/'
label_file_train_reduced = './Fakeddit/train_reduced.csv'

findCorruptImages(train_dir)
train_data = dropUnusedRows(label_file_train, train_dir)
train_data.to_csv(label_file_train)
train_data = removeDatasetBias(label_file_train, train_dir)
train_data.to_csv(label_file_train)

findCorruptImages(train_dir_reduced)
train_data_reduced = dropUnusedRows(label_file_train_reduced, train_dir_reduced)
train_data_reduced.to_csv(label_file_train_reduced)
train_data_reduced = removeDatasetBias(label_file_train_reduced, train_dir_reduced)
train_data_reduced.to_csv(label_file_train_reduced)

validate_dir = './Fakeddit/validate/'
label_file_validate = './Fakeddit/validate.csv'
validate_dir_reduced = './Fakeddit/validate_reduced/'
label_file_validate_reduced = './Fakeddit/validate_reduced.csv'

findCorruptImages(validate_dir)
validate_data = dropUnusedRows(label_file_validate, validate_dir)
validate_data.to_csv(label_file_validate)
validate_data = removeDatasetBias(label_file_validate, validate_dir)
validate_data.to_csv(label_file_validate)

findCorruptImages(validate_dir_reduced)
validate_data_reduced = dropUnusedRows(label_file_validate_reduced, validate_dir_reduced)
validate_data_reduced.to_csv(label_file_validate_reduced)
validate_data_reduced = removeDatasetBias(label_file_validate_reduced, validate_dir_reduced)
validate_data_reduced.to_csv(label_file_validate_reduced)

test_dir = './Fakeddit/test/'
label_file_test = './Fakeddit/test.csv'
test_dir_reduced = './Fakeddit/test_reduced/'
label_file_test_reduced = './Fakeddit/test_reduced.csv'

findCorruptImages(test_dir)
test_data = dropUnusedRows(label_file_test, test_dir)
test_data.to_csv(label_file_test)

findCorruptImages(test_dir_reduced)
test_data_reduced = dropUnusedRows(label_file_test_reduced, test_dir_reduced)
test_data_reduced.to_csv(label_file_test_reduced)
