import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x_array = np.array(x)

    if (rng == None):
        # rand = np.random.random()
        dropout_pattern = np.random.random(x_array.shape)
    else:
        # rand = rng.random()
        dropout_pattern = rng.random(x_array.shape)
    
    dropout_pattern = np.where(dropout_pattern < (1-p), 1/(1-p), 0)

    output = np.multiply(x, dropout_pattern)

    return(output, dropout_pattern)