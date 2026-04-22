import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    return np.sqrt(np.sum((x_arr - y_arr)**2))