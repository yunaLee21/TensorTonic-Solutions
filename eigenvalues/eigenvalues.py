import numpy as np

def is_valid_matrix(matrix):
    # handle empty matrix
    if (len(matrix) == 0):
        return False

    # only one dim
    if (not isinstance(matrix[0], list)):
        return False
    # handle different sizes across elements
    n = len(matrix[0])
    for i in range(len(matrix)):
        if (len(matrix[i]) != n):
            return False
    return True

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    if (not is_valid_matrix(matrix)):
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