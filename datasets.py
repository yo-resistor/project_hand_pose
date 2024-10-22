import os
from utils import reset_data_folder, random_file_move

# create folder environments for train, validation, and test
# if a certain folder does not exist, create
os.makedirs(name='data', exist_ok=True)

os.makedirs(name='data/train', exist_ok=True)       # folder for train
os.makedirs(name='data/valid', exist_ok=True)       # folder for validation
os.makedirs(name='data/test', exist_ok=True)        # folder for test

labels = ['fist', 'up', 'left', 'down', 'right']
"""
class label 0: fist
class label 1: up
class label 2: left
class label 3: down
class label 4: right
"""
for label in labels:
    os.makedirs(name=f"data/train/{label}", exist_ok=True)
    os.makedirs(name=f"data/valid/{label}", exist_ok=True)
    os.makedirs(name=f"data/test/{label}", exist_ok=True)
    
# reset data folder organization
reset_data_folder()

# randomly move files from train folder to validation and test folders for each label
for label in labels:
    random_file_move(label)
