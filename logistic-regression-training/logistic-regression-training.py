import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
    
def bce_loss(y_true, y_pred):
    epsilon = 1e-9
    y1 = y_true * np.log(y_pred + epsilon)
    y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
    return -np.mean(y1 + y2)

def feed_forward(X, weight, bias):
    z = np.dot(X, weight) + bias
    return _sigmoid(z)

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape # X's row: samples; X's columns: features

    # init parameter
    w = np.zeros(n_features)
    b = 0.0

    # gradient descent
    for _ in range(steps):
        A = feed_forward(X, w, b) # compute the prediction p = \sigma(Xw + b)
        bce_loss(y, A) # Compute loss
        dz = A - y # the vector of prediction errors

        # compute gradient
        dw = (1 / n_samples) * (np.dot(X.T, dz))
        db = (1 / n_samples) * (np.sum(dz))

        # update parameter
        w -= lr * dw
        b -= lr * db

    return (w, b)