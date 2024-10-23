import argparse
import os
import random
import torch
import cv2
import torchvision.transforms as transforms

from model import CNN
from utils import pick_random_file

# TODO: navigation function for results

# list all class labels
labels = ['fist', 'up', 'left', 'down', 'right']
num_classes = len(labels)

# construct the argument parser from command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', type=str, default='random',
                    help="choose class label for inference")
args = vars(parser.parse_args())

# define directory and file path based on the input
if args['label'] == 'random':
    label = random.choice(labels)
else:
    label = args['label']

dir = 'data/test/'
dir_label = os.path.join(dir, label)
file_name = pick_random_file(dir_label)
file_path = os.path.join(dir_label, file_name)

# activate gpu if possible, otherwise cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}\n")

# instantiate the model and load the trained weights
model = CNN(num_classes).to(device)
state = torch.load('results/model.pth', map_location=device)
# ref: https://pytorch.org/docs/stable/generated/torch.load.html
model.load_state_dict(state['model_state_dict'])
model.eval()

# define data tranformer for preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# read and preprocess the image
image = cv2.imread(file_path)
orig_image = image.copy()

# get the ground truth label
true_label = file_path

# convert the format to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)

# add batch dimension
# ref: https://pytorch.org/docs/main/generated/torch.unsqueeze.html
image = torch.unsqueeze(image, 0)

# make an inference using the model
with torch.no_grad():
    outputs = model(image.to(device))
_, output_label = torch.max(outputs, 1)
print(outputs)
print(output_label)
pred_label = labels[int(output_label)]
print(pred_label)

cv2.imshow('Result', orig_image)
cv2.waitKey(0)

# image and new image
# if image_new and image_new != image
# space or n -> next image
# s -> save the image
# q or esc -> close the window