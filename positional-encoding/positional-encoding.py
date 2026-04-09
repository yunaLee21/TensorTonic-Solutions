import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))

    # col_pos = np.array((seq_len, 1))
    # row_fre = np.array((1, d_model//2))

    for k in range(seq_len):
        for i in range(0, d_model//2):
            pe[k][2*i] = np.sin(k / (base**(2*i/d_model)))
            pe[k][2*i+1] = np.cos(k / (base**(2*i/d_model)))

    if (d_model % 2 == 1):
        for k in range(seq_len):
            pe[k][-1] = np.sin(k / (base**(2*(d_model//2)/d_model)))
    
    return pe