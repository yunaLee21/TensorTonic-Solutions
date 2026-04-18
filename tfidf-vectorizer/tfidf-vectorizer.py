import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here

    # 1. Build vocabulary
    vocab = set()

    for sentence in documents:
        tokens = sentence.lower()
        tokens = sentence.split() # ['the', 'cat', 'sat'] (1 sentence)
        vocab.update(token for token in tokens)
    vocab = sorted(vocab) # ['cat', 'dog', 'ran', 'sat', 'the']
    
    # 2. 
    doc = len(documents)
    N = len(vocab)

    # 3. DF counter
    df_counter = Counter()
    for sentence in documents:
        tokens = sentence.split()
        unique_terms = set(token for token in tokens)
        df_counter.update(unique_terms)
    # print(df_counter) # Counter({'the': 3, 'cat': 2, 'sat': 2, 'ran': 1, 'dog': 1})

    # 4. Calculate TF IDF
    tdidf_matrix = np.zeros((doc, N))

    for row, col in enumerate(tdidf_matrix):
        sentence = documents[row]
        counter = Counter(sentence)
        # print(row) # 0 1 2
        # print(col) # [0. 0. 0. 0. 0.],..
        
        tokens = sentence.split()
        counter = Counter(tokens)
        # print(counter) # Counter({'the': 1, 'cat': 1, 'sat': 1})

        for idx, token in enumerate(vocab):
            # idx = 0
            # token = cat
            tf = counter[token] / len(tokens)
            idf = math.log(doc / df_counter[token])
            
            tdidf_matrix[row][idx] = tf * idf

    return tdidf_matrix, vocab
    
