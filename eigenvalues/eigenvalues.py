import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    # handle empty matrix
    if (len(matrix) == 0):
        return None

    # handle different sizes across elements
    if (not isinstance(matrix[0], list)):
        return None
    n = len(matrix[0])
    for i in range(len(matrix)):
        if (len(matrix[i]) != n):
            return None
    
    # handle non-square
    A = np.asarray(matrix)
    # not a matrix
    if (A.ndim != 2):
        return None
    if (A.shape[0] != A.shape[1]):
        return None
    
    eigenvalue = np.linalg.eigvals(A)
    indices = np.lexsort((eigenvalue.real, eigenvalue.imag))
    eigenvalue = eigenvalue[indices]

    return eigenvalue