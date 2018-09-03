## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        self.conv1 = nn.Conv2d(1, 16, 5)    # 220 x 220 x 16
        self.pool1 = nn.MaxPool2d(2)        # 110 x 110 x 16
        self.drop1 = nn.Dropout(p=0.2)
        
        self.conv2 = nn.Conv2d(16, 32, 4)   # 107 x 107 x 32
        self.pool2 = nn.MaxPool2d(2)        # 53 x  53 x 32
        self.drop2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(32, 64, 3)   # 51 x 51 x 64
        self.pool3 = nn.MaxPool2d(2)        # 25 x 25 x 64
        self.drop3 = nn.Dropout(p=0.3)
        
        self.conv4 = nn.Conv2d(64, 128, 3)  # 23 x 23 x 128
        self.pool4 = nn.MaxPool2d(2)        # 11 x 11 x 128
        self.drop4 = nn.Dropout(p=0.3)
        
        # dense layers
        self.fc1 = nn.Linear(11*11*128, 1024)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_drop = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(1024, 136)
        
        
    def forward(self, x):

        ## pass x through conv/pool blocks
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))

        # flatten output before the dense layers
        x = x.view(x.size(0), -1)
        
        # pass throug dense layers
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc3(x)

        return x
