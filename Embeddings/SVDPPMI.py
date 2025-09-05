"""
Code to generate an SVD_PPMI matrix.

Takes a PPMI matrix and returns an SVD_PPMI matrix.
dimensions = 300

Follows the code provided in Roth's NLP class
This file was written with the help of generative AI.
"""
from scipy import sparse
import pandas as pd
from scipy.sparse.linalg import svds
from pathlib import Path
from Embeddings.PPMI import compute_ppmi_matrix

# Global Variables
ppmi_matrix_path = Path("Embeddings/ppmi_matrix.npz")
svd_ppmi_matrix_path = Path("Embeddings/svd_ppmi_matrix.csv")

def compute_svd_ppmi_matrix_actually(n = 300):
    """
    computes an SVD_PPMI matrix of dimension n (based on a preexisting PPMI matrix).
    :param n: int, dimension of the PPMI matrix
    """
    ppmi_matrix = sparse.load_npz(ppmi_matrix_path)

    # Perform truncated SVD, keeping the k largest singular values
    U, S, Vt = svds(ppmi_matrix, k=n, which='LM')
    print('SVD (ChatGPT Version) done')

    # Reverse the order (because svds returns the singular values in ascending order)
    word_embeddings = (U[:, ::-1] + Vt.T[:, ::-1]) / 2 #this adds context embedding to the word embedding

    word_embeddings_df = pd.DataFrame(word_embeddings)
    word_embeddings_df.to_csv(svd_ppmi_matrix_path, index=False)
    print('SVD_PPMI matrix saved')

def compute_svd_ppmi_matrix(corpus_path, n = 300):
    if svd_ppmi_matrix_path.exists():
        print('Embedding already computed')
    elif not ppmi_matrix_path.exists():
        compute_ppmi_matrix(corpus_path)
        compute_svd_ppmi_matrix_actually()
    else:
        print('Computing SVD_PPMI matrix')
        compute_svd_ppmi_matrix_actually()




