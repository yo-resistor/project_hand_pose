import argparse
import os
import random
import torch
import cv2
import torchvision.transforms as transforms

from model import CNN # type: ignore
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

# define directory and file path based on the inputv2.imwrite(f"results/{true_label}.png")

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

# control variable for keyboard input message
instruction = True

# infinite loop for displaying the results
while True:
    # read and preprocess the image
    image = cv2.imread(file_path)
    orig_image = image.copy()

    # get the ground truth label
    true_label = file_path.split('/')[-2]

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
    pred_label = labels[int(output_label)]
    
    # display true label over the image
    cv2.putText(orig_image, 
        f"True: {true_label}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    
    # display predicted label over the image
    cv2.putText(orig_image, 
        f"Pred: {pred_label}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, (0, 0, 255), 2, cv2.LINE_AA
    )
    
    # show details about keyboard inputs for user interface
    if instruction:
        print('\nInstruction:')
        print('\nPress "n" or "space" to see the next result.')
        print('Press "s" to save the current result.')
        print('Press "q" or "esc" to terminate the program.\n')
        instruction = False
        
    # show the result
    print(f"True label: {true_label}, Predicted label: {pred_label}")
    cv2.imshow('Result', orig_image)
    
    # wait for a key press 
    # ref: https://www.asciitable.com/
    key = cv2.waitKey(0)

    # check keyboard input
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        # save image frame
        # if the prediction is correct, save in 'correct' folder
        if true_label == pred_label:
            n_files = len(os.listdir('results/correct'))
            cv2.imwrite(f"results/correct/{n_files}_{true_label}.png", orig_image)
        # if the prediction is wrong, save in 'wrong' folder
        else:
            n_files = len(os.listdir('results/wrong'))
            cv2.imwrite(f"results/wrong/{n_files}_{true_label}_{pred_label}.png", orig_image)
        
    elif key == ord('n') or key == 32:
        # select next image based on the label the user chose
        if args['label'] == 'random':
            label = random.choice(labels)
        else:
            label = args['label']
        
        # define new image file path
        dir_label = f"data/test/{label}/"
        file_name = pick_random_file(dir_label)
        file_path = os.path.join(f"data/test/{label}/", file_name)

# close all windows
cv2.destroyAllWindows()