import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    return np.sum(np.abs(x_arr - y_arr))