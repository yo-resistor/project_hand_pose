import torch
import matplotlib.pyplot as plt
import os
import shutil
import random

# function to save the trained model in local environment
def save_model(epochs, model, optimizer, criterion):
    """
    ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/model.pht')
    
# function to save the loss and accuracy plots in local environment
def save_plot(train_acc, valid_acc, train_loss, valid_loss):
    """
    ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    # define the style of plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # accuracy plot
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='Train accuracy')
    plt.plot(valid_acc, color='blue', linestyle='-', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/accuracy.png')
    
    # loss plot
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='green', linestyle='-', label='Train loss')
    plt.plot(valid_loss, color='blue', linestyle='-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/loss.png')

# function to reset data folder organizations
def reset_data_folder():
    """
    run this function before collecting data
    this function resets the folder environment for data by
    moving all data from valid and test folders to train folder
    """
    # define the sources and destination folders
    folder_valid = 'data/valid/'    # source 1
    folder_test = 'data/test/'      # source 2
    folder_train = 'data/train/'    # destination

    # list of subfolders (fist, up, down, right, left)
    subfolders = ['fist', 'up', 'down', 'right', 'left']
    
    # loop through each subfolder in validation folder
    for subfolder in subfolders:
        # define the full path for each subfolder
        subfolder_valid = os.path.join(folder_valid, subfolder)
        subfolder_test = os.path.join(folder_test, subfolder)
        subfolder_train = os.path.join(folder_train, subfolder)
        
        # ensure all subfolders exist, if not, create
        if not os.path.exists(subfolder_valid):
            os.makedirs(subfolder_valid)
        if not os.path.exists(folder_test):
            os.makedirs(folder_test)
        if not os.path.exists(subfolder_train):
            os.makedirs(subfolder_train)
            
        # move files only if they exist in the subfolder
        if os.listdir(subfolder_valid):
            move_files(src=subfolder_valid, dst=subfolder_train)
        else:
            print(f"No files have been moved from {subfolder_valid}.")
        if os.listdir(subfolder_test):
            move_files(src=subfolder_test, dst=subfolder_train)
        else:
            print(f"No files have been moved from {subfolder_test}.")
        
# function to move files from source to destination
def move_files(src: str, dst: str):
    # initialize counter for the number of moved files
    count = 0
    
    # loop through each file in the source
    for file_name in os.listdir(src):
        # define the full file paths
        src_file = os.path.join(src, file_name)
        dst_file = os.path.join(dst, file_name)
        
        # move the file from the source to the destination
        shutil.move(src_file, dst_file)
        count += 1
    
    # print out result
    print(f"{count} files have been moved from {src} to {dst}.")
    
# function to randomly move files from source to destination by a specific ratio
def random_file_move(label_name: str, ratio=0.15):
    """
    label_name = fist, up, left, down, right
    train data will be around 70% of data
    test and validation data will be around 15% of data each
    """
    # define full file paths
    src = f"data/train/{label_name}/"
    dst_valid = f"data/valid/{label_name}/"
    dst_test = f"data/test/{label_name}/"
    
    # get file list in source folder
    files_src = os.listdir(src)
    
    # define the number of files to move to each validation or test folder
    n_files_to_move = round(len(files_src) * ratio)
    if n_files_to_move < 1:
        print("There are no files to move.")
        return
    
    # select random files in source folder and move
    # total number of files to move is 2 * number of files to move to each folder
    files_to_move = random.sample(files_src, 2 * n_files_to_move)
    
    # move the selected files to each folder
    # 1st half to validation folder and 2nd half to test folder
    for file_name in files_to_move[0:n_files_to_move]:
        shutil.move(os.path.join(src, file_name), dst_valid)
    for file_name in files_to_move[n_files_to_move:]:
        shutil.move(os.path.join(src, file_name), dst_test)