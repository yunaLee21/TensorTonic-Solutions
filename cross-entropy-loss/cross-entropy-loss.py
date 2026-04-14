import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    idx = np.arange(len(y_true)) #[0, 1, 2, 3,...] -> for indexing
    loss = y_pred[idx, y_true] # extract the value at the correct label
    
    return - np.mean(np.log(loss))