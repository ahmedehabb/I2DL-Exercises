"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        
        #After convolution --> image shape = ((N-F)/S) + 1
        #After Pooling --> image shape = ((N-F)/S) + 1

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1)  # Convolution2d1
        nn.init.xavier_uniform_(self.conv1.weight)
        # now image shape: (96 - 4)/1 + 1 = 93
        self.activation1 = nn.ELU()  # Activation1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Maxpooling2d1
        self.dropout1 = nn.Dropout(p=0.1)  # Dropout1
        # now image shape: (93 - 2)/2 + 1 = 46.5 -> 46

        # so image shape : 46*46*32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # Convolution2d2
        nn.init.xavier_uniform_(self.conv2.weight)

        # now image shape: (46 - 3)/1 + 1 = 44
        self.activation2 = nn.ELU()  # Activation2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Maxpooling2d2
        self.dropout2 = nn.Dropout(p=0.2)  # Dropout2
        # now image shape: (44 - 2)/2 + 1 = 22

        # 22*22*64

        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1)  # Convolution2d3
        nn.init.xavier_uniform_(self.conv3.weight)
        
        # now image shape: (22 - 1)/1 + 1 = 21
        self.activation3 = nn.ELU()  # Activation3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Maxpooling2d3
        self.dropout3 = nn.Dropout(p=0.3)  # Dropout3
        # now image shape: (21 - 2)/2 + 1 = 10.5 -> 10

        # 10*10*128

        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1)  # Convolution2d4
        nn.init.xavier_uniform_(self.conv4.weight)
        
        # now image shape: (10 - 1)/1 + 1 = 10
        self.activation4 = nn.ELU()  # Activation4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Maxpooling2d4
        self.dropout4 = nn.Dropout(p=0.4)  # Dropout4
        # now image shape: (10 - 2)/2 + 1 = 5

        # 5*5*256

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 5 * 5, 688)
        self.activation5 = nn.ELU()
        self.dropout5 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(688, 688)
        self.activation6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.6)


        self.fc3 = nn.Linear(688, 30)  # Dense3

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.dropout1(self.maxpool1(self.activation1(self.conv1(x))))
        x = self.dropout2(self.maxpool2(self.activation2(self.conv2(x))))
        x = self.dropout3(self.maxpool3(self.activation3(self.conv3(x))))
        x = self.dropout4(self.maxpool4(self.activation4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout5(self.activation5(self.fc1(x)))
        x = self.dropout6(self.activation6(self.fc2(x)))
        x = self.fc3(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
