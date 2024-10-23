import torch.nn as nn
import torch.nn.functional as F

# define CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        # define pooling layer
        # max pool with (2, 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # define convolutional layers
        # convolutional layer 1 -> input: (3, 360, 640), output: (16, 180, 320) after pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # convolutional layer 2 -> input: (16, 180, 320), output: (32, 90, 160) after pooling
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # convolutional layer 3 -> input: (32, 90, 160), output: (64, 45, 80) after pooling
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # convolutional layer 4 -> input: (64, 45, 80), output: (128, 22, 40) after pooling
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # define dropout layer after convolutional layers
        self.dropout_conv = nn.Dropout(p=0.2)
        
        # define fully connected layers
        # fully connected layer 1 -> flattened the input to 512 neurons
        self.fc1 = nn.Linear(128 * 22 * 40, 512)
        # fully connected layer 2 -> 512 to number of classes 
        self.fc2 = nn.Linear(512, num_classes)
        
        # define dropout layer after fully connected layers
        self.dropout_fc = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # apply convolutional layer -> relu -> pooling
        x = self.pool(F.relu(self.conv1(x)))    # (3, 360, 640) -> (16, 180, 320)
        x = self.pool(F.relu(self.conv2(x)))    # (16, 180, 320) -> (32, 90, 160)
        x = self.pool(F.relu(self.conv3(x)))    # (32, 90, 160) -> (64, 45, 80)
        x = self.pool(F.relu(self.conv4(x)))    # (64, 45, 80) -> (128, 22, 40)
        
        # apply dropout layer after convolutional layers
        x = self.dropout_conv(x)
        
        # flatten the output from convolutional layers
        x = x.view(-1, 128 * 22 * 40)
        
        # apply fully connected layers -> relu -> dropout
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        # return the result
        return x