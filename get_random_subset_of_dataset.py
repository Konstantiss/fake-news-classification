import os
import random
import shutil


def subset(src_dir, trg_dir, num_images):
    file_list = os.listdir(src_dir)
    random.shuffle(file_list)
    files_to_copy = file_list[0:num_images]

    for file in files_to_copy:
        file_path = os.path.join(src_dir, file)
        shutil.copy(file_path, trg_dir)

train_dir = './Fakeddit/train/'
reduced_train_dir = './Fakeddit/train_reduced'
subset(train_dir, reduced_train_dir, 15000)

valid_dir = './Fakeddit/validate/'
reduced_valid_dir = './Fakeddit/validate_reduced'
subset(valid_dir, reduced_valid_dir, 2000)

test_dir = './Fakeddit/test/'
reduced_test_dir = './Fakeddit/test_reduced'
subset(test_dir, reduced_test_dir, 4000)

