# About: This file defines a 2-layer neural network that learns a transformation of the unit disk, D as defined in utils.py. 
#        The neural network is to preserve the initial area of the region (as given by all transformed points). 


# Importing PyTorch's tensor operations and neural network module. 
import torch 
import torch.nn as nn

# Defining a new neural network architecture designed to learn a map that mimics symplectic maps in 2D i.e. area preservation. 
# Essentially a function f_θ: R^2 → R^2, where θ are the trainable parameters of this neural network, that is trained to satisfy |det(Jacobian)| ≈ 1. 
class area_preserving_NN(nn.Module):

    # Constructor. 
    def __init__(self): 

        # Calling parent class constructor. 
        super(area_preserving_NN, self).__init__() 
        
        # Creating the attribute 'net', the core function f_θ (transformation rule) to hold a sequence of operations.
        self.net = nn.Sequential(
            nn.Linear(2, 64), # Input layer: R^2 → R^64. 
            nn.Tanh(), # Activation function that allows non-linear mappings. 
            nn.Linear(64, 64), # Hidden layer. 
            nn.Tanh(), # Activation again. 
            nn.Linear(64, 2) # Output layer: gives the transformed point (x', y') in R^2. 
        )
    
    # The forward method executes the network's transformation. This becomes the learned 
    # mapping f(x, y) = (x', y'). 
    def forward(self, x):  
        return self.net(x) # Applying the sequential operations defined under 'self.net' to the input 'x'.  
