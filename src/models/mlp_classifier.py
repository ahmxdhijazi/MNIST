"""
Goals:
Requires at least one hidden layer, so:
    An input layer (784 -> 256)
    A non-linear activation (using nn.ReLU)
    A hidden layer (256 -> 128)
    Another nn.ReLU
    An output layer (128 -> 10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self):
        #Initializes the MLP model by defining its layers.

        super(MLPClassifier, self).__init__()

        # Define required layers
        self.fc1 = nn.Linear(784, 256)  #Input to Hidden Layer 1
        self.fc2 = nn.Linear(256, 128)  #Hidden Layer 1 to Hidden Layer 2
        self.fc3 = nn.Linear(128, 10)  #Hidden Layer 2 to Output

    def forward(self, x): #Defines the forward pass, including the non-linear activations.

        #Flatten the input image
        x = x.view(x.size(0), -1)

        #Pass through first layer and apply activation
        x = self.fc1(x)
        x = F.relu(x)  #non-linearity

        #Pass through second layer and apply activation
        x = self.fc2(x)
        x = F.relu(x)

        #Pass through the final output layer
        out = self.fc3(x) #loss function will handle applying activation

        return out

"""
Temporary Testing block with random temp data so we don't explode later.
if __name__ == '__main__':
    model = MLPClassifier()
    print("Model Architecture:")
    print(model)

    temp_input = torch.randn(64, 1, 28, 28)
    output = model(temp_input)

    print(f"\nShape of input: {temp_input.shape}")
    print(f"Shape of output: {output.shape}")
"""