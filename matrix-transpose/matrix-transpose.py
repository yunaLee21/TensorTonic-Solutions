import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A_matrix = np.array(A)
    h, w = A_matrix.shape
    A_T = np.zeros((w, h))

    for i in range(w):
        for j in range(h):
            A_T[i][j] = A_matrix[j][i]

    return A_T