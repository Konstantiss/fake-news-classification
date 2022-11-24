import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np
import sys

parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')

parser.add_argument('type', type=str, help='train, validate, or test')

args = parser.parse_args()

df = pd.read_csv(args.type, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

pbar = tqdm(total=len(df))

cur_dir = os.getcwd()

# if not os.path.exists("images"):
#   os.makedirs("images")
# for index, row in df.iterrows():
#   if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
#     image_url = row["image_url"]
#     try:
#       response = urllib.request.urlretrieve(image_url, "images/" + row["id"] + ".jpg")
#     except:
#       continue
for index, row in df.iterrows():
  if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    file_list = os.listdir(cur_dir)
    file_name = row["id"] + ".jpg"
    if file_name in file_list:
        print("File already exists")
        continue
    try:
      response = urllib.request.urlretrieve(image_url, "./" + row["id"] + ".jpg")
    except:
      continue
  pbar.update(1)
print("done")
