import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import CNN
from datasets_inference import train_loader, valid_loader, test_loader
from utils import save_model, save_plot

# TODO: confusion matrix

# construct the argument parser from command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help="the number of epochs to train the model for")
args = vars(parser.parse_args())

# define learning parameters
lr = 1e-3
epochs = args['epochs']

# find the number of classes
num_classes = len(os.listdir('data/train'))

# activate gpu if possible, otherwise cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}\n")

# instantiate and send the model to the device
model = CNN(num_classes).to(device)
print(model)

# model's details: total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params}")
print(f"Total trainable parameters: {total_train_params}\n")

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# define loss function
criterion = nn.CrossEntropyLoss()

## training
# define a method for training per epoch
def train(model, optimizer, criterion, train_loader):
    # train mode
    model.train()
    
    # define variables to plot the results
    train_loss = 0.0        # for loss per epoch
    train_correct = 0       # for accuracy per epoch
    total = 0
    
    # initialize tqdm for training loop
    train_bar = tqdm(train_loader, desc='Training')
    
    # perform training using a train loader
    for inputs, labels in train_bar:
        # send data to the device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients to make sure they are zero
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # backpropagation 
        loss.backward()
        
        # optimization
        optimizer.step()
        
        # compute training statistics
        train_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        train_correct += (predictions == labels).sum().item()
        
    # loss and accuracy per epoch
    epoch_loss = train_loss / total
    epoch_acc = 100. * train_correct / total
    
    # return the results
    return epoch_loss, epoch_acc

## validation
# define a method for validation per epoch
def validate(model, criterion, valid_loader):
    # validation mode
    model.eval()
    
    # define variables to plot the results
    valid_loss = 0.0        # for loss per epoch
    valid_correct = 0       # for accuracy per epoch
    total = 0
    
    # initialize tqdm for validation loop
    valid_bar = tqdm(valid_loader, desc='Validation')
    
    # perform training using a valid loader
    # show the progress using tqdm
    with torch.no_grad():
        # disable gradient descent calculation for evaluation
        for inputs, labels in valid_bar:
            # send data to the device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # compute training statistics
            valid_loss += loss.item()
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            print(labels)
            print(predictions)
            valid_correct += (predictions == labels).sum().item()
    
    # loss and accuracy per epoch
    epoch_loss = valid_loss / total
    epoch_acc = 100. * valid_correct / total
    
    # return the results
    return epoch_loss, epoch_acc

# define container lists to store losses and accuracies
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []

# run training and validation methods in epoch
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_epoch_loss, train_epoch_acc = train(model, optimizer, criterion, train_loader)
    valid_epoch_loss, valid_epoch_acc = validate(model, criterion, valid_loader)
    
    # store the results in the container
    train_loss.append(train_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_loss.append(valid_epoch_loss)
    valid_acc.append(valid_epoch_acc)
    
    # print the results per epoch
    print(f"Training loss: {train_epoch_loss:.4f}, Training accuracy: {train_epoch_acc:.2f}%")
    print(f"Validation loss: {valid_epoch_loss:.4f}, Validation accuracy: {valid_epoch_acc:.2f}%\n")

# save the trained model and weights
save_model(epochs=epochs, model=model,
           optimizer=optimizer, criterion=criterion)

# save the loss and accuracy plots
save_plot(train_acc= train_acc, valid_acc=valid_acc, 
          train_loss=train_loss, valid_loss=valid_loss)

# print that the training is done
print(f"TRAINING COMPLETE")

#### TEST ####
