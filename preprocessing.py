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

def findCorruptImages():
    # Create a pool of processes to check images
    p = Pool()

    # Create a list of images to process
    files = [f for f in glob.glob("./Fakeddit/train/*.jpg")]

    print(f"Files to be checked: {len(files)}")

    # Map the list of files to check onto the Pool
    result = p.map(checkSingleImage, files)

    # Filter out None values representing files that are ok, leaving just corrupt ones
    result = list(filter(None, result))
    print(f"Num corrupt files: {len(result)}")

def dropUnusedRows(annotations_file, img_dir):
    dataset = pd.read_csv(annotations_file, sep='\t')
    file_list = os.listdir(img_dir)
    file_list = {file.replace('.jpg', '') for file in file_list}
    dataset = dataset[dataset.id.isin(file_list)]
    return dataset

# findCorruptImages()

train_dir = './Fakeddit/train/'
label_file_train = './Fakeddit/all_train.tsv'
train_data = dropUnusedRows(label_file_train, train_dir)
train_data.to_csv('./Fakeddit/train_reduced.csv')

test_dir = './Fakeddit/test/'
label_file_test = './Fakeddit/all_test_public.tsv'
test_data = dropUnusedRows(label_file_test, test_dir)
test_data.to_csv('./Fakeddit/test_reduced.csv')

