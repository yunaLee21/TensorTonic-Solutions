import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y_arr = np.array(y)

    values, counts = np.unique(y_arr, return_counts=True)

    # [1,1,2,2,2,3,4]
    # print(values) # [1 2 3 4]
    # print(counts) # [2 3 1 1]
    counts = np.array(counts)
    n = len(y)

    p = counts / n
    # print(p) #[0.28571429 0.42857143 0.14285714 0.14285714]

    result = - (p * np.log2(p))

    return np.sum(result)

    