"""
__init__(self): The constructor. This is where I will define all the layers the model will use. Just one nn linear layer
forward(self, x): Method defines how data flows through the model.
    says "First, the data goes here, then it goes here..."
y = Wx, which is a single linear layer.
input : (x) will be a flattened image with 28*28 = 784 features.
output : (y) need one for each digit (0-9), 10 total
"""

import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self):
        #Defining its layers
        super(LinearClassifier, self).__init__()
        self.linear_layer = nn.Linear(in_features=784, out_features=10) #Mentioned above, 784 features in, 10 output features

    def forward(self, x): #forward pass, how the input becomes the output, data flowing forward through the model
        #flatten the input image
        #   x starts as shape: [batch_size, 1, 28, 28]
        #   We need to reshape it to: [batch_size, 784]
        #   get batch size and automatically calculate the remaining dimension using -1 (which will be 784).
        x = x.view(x.size(0), -1)

        #pass flattened data through the linear layer
        out = self.linear_layer(x)

        return out

#Test functionality with temporary data!
if __name__ == '__main__':
    # Create an instance of the model
    model = LinearClassifier()
    print(f"Model Architecture: {model}\n")

    #creating a temporary batch of data (64 images, 1 channel, 28x28) for testing.
    temp_input = torch.randn(64, 1, 28, 28)

    # Pass the temp data through the model before we try the MNIST data
    output = model(temp_input)

    print(f"Shape of input: {temp_input.shape}")
    print(f"Shape of output: {output.shape}")