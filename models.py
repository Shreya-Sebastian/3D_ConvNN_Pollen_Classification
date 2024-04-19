import torch
import torch.nn as nn
import torchvision.models as models


class ConvNet3D(nn.Module):
    def __init__(self, param):
        super(ConvNet3D, self).__init__()
        
        in_channels = param.num_frames

        # Define the 3D convolution layers
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)

        # Define the pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # Define the dropout layer
        self.dropout = nn.Dropout(0.2)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 112 * 112, 128) 
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x):
        
        # Apply 3D convolutions and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 1 * 112 * 112)

        # Apply fully connected layers
        x = self.dropout(x)                                 
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits     

class ResNet3D(nn.Module):
    def __init__(self, param):
        super(ResNet3D, self).__init__()

        in_channels = param.num_frames
        
        # Import ResNet3D model without pretrained weights
        resnet3d = models.video.r3d_18(pretrained=False)
        # Initialize first layer of ResNet architecture
        resnet3d.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2))
        self.features = resnet3d
        # Define output layer 
        self.fc = nn.Linear(400, 3)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
