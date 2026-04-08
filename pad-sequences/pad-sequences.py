import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    L = max_len if (max_len != None) else max((len(seq) for seq in seqs), default=0)

    pad_seq = np.full((N, L), pad_value)

    for i, seq in enumerate(seqs):
        valid_value = min(len(seq), L)
        pad_seq[i, :valid_value] = seq[:valid_value]

    return pad_seq