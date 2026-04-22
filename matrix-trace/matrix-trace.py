import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A_matrix = np.array(A)
    trace = 0
    for i in range(len(A)):
        trace += A_matrix[i][i]

    return trace