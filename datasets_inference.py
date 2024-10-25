import os
from utils import random_file_move
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# define batch_size
BATCH_SIZE = 8

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

# randomly move files from train folder to validation and test folders for each label
for label in labels:
    random_file_move(label)
    
# define data transformer for training and validation
# ref: https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html
# training transforms
train_transform = transforms.Compose([
    # define gaussian blur transformation with a (5, 5) kernel and sigma range of (0.1, 2.0)
    # ref: https://pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    # ensure image size to be 360 (height) X 640 (width)
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# validation transforms
valid_transform = transforms.Compose([
    # ensure image size to be 360 (height) X 640 (width)
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# test transforms
test_transform = transforms.Compose([
    # ensure image size to be 360 (height) X 640 (width)
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

## datasets
# training dataset
train_dataset = datasets.ImageFolder(
    root='data/train',
    transform=train_transform
)

# validation dataset
valid_dataset = datasets.ImageFolder(
    root='data/valid',
    transform=valid_transform
)

# test dataset
test_dataset = datasets.ImageFolder(
    root='data/test',
    transform=test_transform
)

# DataLoader
# ref: https://pytorch.org/docs/stable/data.html
# training data loader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory= True
)

# validation data loader
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# test data loader
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)