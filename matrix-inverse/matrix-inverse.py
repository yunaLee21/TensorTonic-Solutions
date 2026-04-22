import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A_matrix = np.array(A)
    
    if (A_matrix.ndim != 2):
        return None
    if (A_matrix.shape[0] != A_matrix.shape[1]):
        return None
    threshold = 1e-10
    if (abs(np.linalg.det(A_matrix)) < threshold):
        return None

    # print(A_matrix.ndim)
    # print(A_matrix.shape[0])
    # print(A_matrix.shape[1])
    # print(np.linalg.det(A_matrix))
    A_inv = np.linalg.inv(A_matrix)

    return A_inv