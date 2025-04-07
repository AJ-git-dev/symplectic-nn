# About: This file defines utility functions for generating 2D point samples from the unit disk, 
# computing the Jacobian determinant of a neural map f: R^2 → R^2, and evaluating a loss function 
# that measures how far the learned transformation is from being locally area-preserving.


import numpy as np # Used to generate random points in the unit disk. 
import torch # Used for gradient-based training. 
import matplotlib.pyplot as plt # Used to visualize the shapes before and after the transformation. 

# Function purpose: constructs a set of n 2D points that lie on the unit disk given by D = {(x, y) ∈ R^2 | x^2 + y^2 ≤ 1}.
#                   Mathematically, the polar coordinate system is used. 
# Function parameter(s): integer quantity of points inside D. n_points = 1000 means 1000 points in D by default (change to your liking). 
# Function return value(s): returns a NumPy array of shape (n_points, 2), each row is a 2D point within D. 
def generate_unit_disk(n_points=1000):

    # Returns a 1D NumPy array of shape 'n_points' containing float-valued angles randomly sampled on the interval [0, 2 * np.pi) radians.
    # Mathematically, this corresponds to establishing the angular position of each sampled point.  
    theta = np.random.uniform(0, 2 * np.pi, n_points) 
                                            
    # Returns a 1D NumPy array of shape 'n_points' containing float-valued radii randomly sampled on the interval [0, 1) units. 
    # The square root ufunc sets the sample radii such that PDF p_r(r) = 2r and dA = rdrdθ. 
    r = np.sqrt(np.random.uniform(0, 1, n_points)) 
                                                   
    # Using 'theta' and 'r' to determine the Cartesian coordinates 'x' and 'y' for each point. 
    x = r * np.cos(theta) 
    y = r * np.sin(theta)

    # Returning a 2-columned matrix of sampled point data to pass to the neural network. 
    return np.stack([x, y], axis=1) 


# Function purpose: This function computes the Jacobian of the mapping f: R^2 → R^2 for each input point.
# Function parameter(s): 'f' is the neural network representing the function f. 
#                        'x' is a PyTorch tensor of shape '(batch_size, 2)' where each row represents a point in R^2.
# Function return value(s): returns the vector of determinant values to be used in the loss function. 
def jacobian_determinant(f, x):
    
    # Automatic differentiation on each row in 'x'. 
    x.requires_grad_(True) 

    y = f(x) # Shape (batch_size, 2). 

    # Initializing an empty list to hold partial gradients for each output coordinate. J will hold both rows of the Jacobian matrix across the batch. 
    J = []

    # Looping over the two output dimensions, 'f_1' and 'f_2', to compute the gradient of each coordinate w.r.t. the input vector. 
    for i in range(2): 
        grad = torch.autograd.grad(
            outputs=y[:, i], # Selects the ith component of the output vector for every point in the batch. 
            inputs=x, # Taking the derivative w.r.t x. 
            grad_outputs=torch.ones_like(y[:, i]), 
            retain_graph=True, 
            create_graph=True # Building a computation graph for the gradient. 
        )[0] 
        J.append(grad) 

    jacobian = torch.stack(J, dim=2) # Combining the two gradient arrays into a single Jacobian tensor. 
    det = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0] # Taking the determinant. 
    return det

# Function purpose: This function computes a loss value which quantifies how much area is not preserved.
# Function parameter(s): 'model' is a neural network parameter, to which 'f' is passed. 
#                        'x' is a PyTorch tensor of shape (batch_size, 2) to which 'disk' is passed in main.ipynb. 
# Function return value(s): The mean squared deviation of the Jacobian matrix determinant from 1 across the entire input batch to measure 
#                           how far the map is from being locally area-preserving; returns 0 if the area is perfectly preserved, non-0 for contraction/expansion. 
def area_preservation_loss(model, x):

    # Storing the return value of 'jacobian_determinant().' 
    det = jacobian_determinant(model, x)

    return ((det - 1) ** 2).mean()
