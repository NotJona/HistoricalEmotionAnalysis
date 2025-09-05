"""
Code to generate a PPMI matrix.
Takes a corpus in the form of a list of lists with each list representing a document in the corpus.
Returns a co-occurrence matrix and a PPMI matrix

This file was written with the help of generative AI.
"""
from collections import defaultdict, Counter
import numpy as np
import json
from scipy import sparse
from scipy.sparse import lil_matrix
from pathlib import Path

# Global Variables
co_occurrence_matrix_path = Path("Embeddings/co_occurrence_matrix.npz")
unique_words_path = Path("Embeddings/AAA_word_list.txt")
ppmi_matrix_path = Path("Embeddings/ppmi_matrix.npz")

def co_occurrence_matrix(corpus_path, window_size=4):
    """
    takes the path to a corpus file (.json) and computes a co-occurrence matrix
    :param corpus_path: str, path to the corpus file
    :param window_size: int, size of the context window (default 4 following Hellrich et al.)
    """
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    unique_words = []

    # Create a list of co-occurring word pairs
    co_occurrences = defaultdict(Counter)
    count = 1

    for words in corpus:
        count += 1
        for i, word in enumerate(words):
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    co_occurrences[word][words[j]] += 1
        unique_words = list(set(unique_words+words))

    # Initialize the co-occurrence matrix
    co_matrix = lil_matrix((len(unique_words), len(unique_words)), dtype=int)
    print('empty matrix successfully initialized')

    # Populate the co-occurrence matrix
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word], word_index[neighbor]] = count
    print('co-occurrence matrix successfully generated')

    #save the co-occurrence matrix
    sparse_ppmi = sparse.csr_matrix(co_matrix)
    sparse.save_npz(co_occurrence_matrix_path, sparse_ppmi)
    print('co-occurrence matrix successfully saved')

    with open(unique_words_path, "w") as file:
        for word in unique_words:
            file.write(f"{word}\n")
    print('word_list successfully saved')

    return co_matrix

def compute_ppmi_matrix_actually(corpus_path, window_size=4):
    """
    takes the path to a corpus file (.json) and computes a PPMI matrix
    :param corpus_path: str, path to the corpus file
    :param window_size: int, size of the context window (default 4 following Hellrich et al.)
    """
    matrix = co_occurrence_matrix(corpus_path)
    matrix = matrix.tocsr()

    total_count = matrix.sum() #|D|
    sum_over_rows = np.array(matrix.sum(axis=1)).flatten() # #(w)
    sum_over_cols = np.array(matrix.sum(axis=0)).flatten() # #(c)

    if 0 in sum_over_rows or 0 in sum_over_cols:
        return "There is a 0 row or column in your co-occurrence matrix. ERROR!"

    # Initialize the ppmi-matrix
    ppmi = sparse.lil_matrix(matrix.shape)
    print('ppmi matrix successfully initialized')

    # Iterate only over non-zero entries of co_matrix
    coo = matrix.tocoo()
    for i, j, count in zip(coo.row, coo.col, coo.data):
        p_w = sum_over_rows[i]
        p_c = sum_over_cols[j]
        p_wc = count

        #Cause issues with datatypes
        p_wc = np.float64(p_wc)
        total_count = np.float64(total_count)
        p_w = np.float64(p_w)
        p_c = np.float64(p_c)

        p = p_wc * total_count / (p_w * p_c)
        if p > 1: #since PPMI sets values log(x) <= 0 to 0, which corresponds to x <= 1
            ppmi[i, j] = np.log(p)

    print('ppmi matrix successfully generated')
    sparse.save_npz(ppmi_matrix_path, ppmi.tocsr())

def compute_ppmi_matrix(corpus_path, window_size=4):
    """
    takes the path to a corpus file (.json) and computes a PPMI matrix
    :param corpus_path: str, path to the corpus file
    :param window_size: int, size of the context window (default 4 following Hellrich et al.)
    """
    if ppmi_matrix_path.exists():
        print('Embedding already computed')
    else:
        print('Computing PPMI matrix ...')
        compute_ppmi_matrix_actually(corpus_path, window_size)














