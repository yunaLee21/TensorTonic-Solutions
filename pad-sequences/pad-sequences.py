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

    pad_seq = np.zeros((N, L))

    for i, seq in enumerate(seqs):
        seq_len = len(seq)
        padding_needed = max(0, L - seq_len)
        truncation_needed = max(0, seq_len - L)
        
        padding_array = np.array([pad_value] * padding_needed)
        seq = seq[:-truncation_needed] if truncation_needed > 0 else seq

        pad_seq[i] = np.concatenate((seq, padding_array))

    return pad_seq