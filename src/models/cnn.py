import torch.nn as nn
import torch.nn.functional as F #module for 'relu' that don't have weights


class CNNClassifier(nn.Module):
    def __init__(self): #constructor
        #Initializes the CNN model: Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> FC
        super(CNNClassifier, self).__init__()

        #Convolutional Layers
        #Input: 1 channel (grayscale), Output: 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3) # Kernel: 3x3 square, aka 3x3 sliding window
        #Input: 32 channels (from conv1), Output: 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) #Accepts 32, and creates another 64 more feature maps

        #Max pooling layer to shrink the image while keeping strongest features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        """
        calculate the flattened size after pooling for 28x28
        After conv1 (3x3): 26x26 (28-2)
        After pool (2x2): 13x13 (26/2)
        After conv2 (3x3): 11x11 (13-2)
        After pool (2x2): 5x5 (11//2, rounds down)
        Final flattened size = 64 (channels) * 5 * 5 = 1600
        """
        self.fc1 = nn.Linear(64 * 5 * 5, 128) #1600 inputs into 128 AKA the hidden layer
        self.fc2 = nn.Linear(128, 10)  # Output layer, 128 -> 10, score for 0-9

    def forward(self, x): #Forward pass

        #First Conv Block
        x = self.conv1(x) #first conv layer
        x = F.relu(x) #apply ReLu activation function to add non-linearity
        x = self.pool(x) #pass through pooling layer to shrink

        #Second Conv Block
        x = self.conv2(x) #second conv layer
        x = F.relu(x)     #apply ReLu
        x = self.pool(x)  #pool

        #Flatten for the Linear Layers, flatten all dimensions except the batch size
        x = x.view(x.size(0), -1) #2D feature maps into a 1D vector to feed the linear layers

        #Pass through Linear Layers
        x = self.fc1(x) #pass the features through first linear layer
        x = F.relu(x)
        out = self.fc2(x) #pass the remaining features through second linear layer

        return out #returns the final 10 scores


""""--- Test block ---
if __name__ == '__main__':
    model = CNNClassifier()
    print("Model Architecture:")
    print(model)

    temp_input = torch.randn(64, 1, 28, 28)
    output = model(temp_input)

    print(f"\nShape of input: {temp_input.shape}")
    print(f"Shape of output: {output.shape}")
"""