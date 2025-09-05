"""
This file implements Random Walk

Written with the help of GenAI
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize
from scipy.sparse import diags, csr_matrix
from functions import load_VAD_seed_lexion, save_as_VAD, load_seed_data, load_GOLD_data, get_gold_words


def retrieve_embeddings(embedding_name, lexicon_path, gold_standard_path):
    """
    returns dict of form word: word_vector for all words in the COHA
    :param embedding_name: str, name of embedding
    :param lexicon_path: str, path to seed word lexicon
    :param gold_standard_path: str, path to gold standard lexicon
    :return: dict of form word: word_vector
    """
    seed_data = load_seed_data(embedding_name, lexicon_path)
    GOLD_data = load_GOLD_data(embedding_name, gold_standard_path)
    word_vector_dict = dict()
    for element in seed_data:
        word_vector_dict[element[0]] = element[2]
    for row in GOLD_data:
        word_vector_dict[row[0]] = row[1]
    return word_vector_dict

def random_walk(embedding_name, lexicon_path, gold_standard_path, vad_lexicon_name, save_folder_name,
                beta=0.9, max_iter=100, n_neighbors=25):
    """
    performs a random walk to induce historical VAD scores. Saves result under:
    'HistoricalVAD/RandomWalk/vad_lexicon_name _ model_name _ historicalVAD.tsv'
    :param embedding_name: str, name of the embedding
    :param lexicon_path: str, path to seed word lexicon
    :param gold_standard_path: str, path to gold standard lexicon
    :param vad_lexicon_name: str, name of the seed word lexicon (important for saving)
    :param save_folder_name: str, name of the saved folder
    :param beta: float, weight term
    :param max_iter: int, maximum number of iterations
    :param n_neighbors: int, number of neighbors
    """
    word_embeddings = retrieve_embeddings(embedding_name, lexicon_path, gold_standard_path)
    seed_vad_scores = load_VAD_seed_lexion(lexicon_path)

    # Step 0: Prepare data and graph
    words = list(word_embeddings.keys())
    embeddings = np.array([word_embeddings[w] for w in words])
    n_words = len(words)
    word_to_idx = {word: i for i, word in enumerate(words)}

    # Step 1: Build graph with angular distance weights
    cosine_sim = cosine_similarity(embeddings)

    # Create edge weight matrix E using angular distance for top neighbors
    E = np.zeros_like(cosine_sim)
    for i in range(n_words):
        # Get top n_neighbors by cosine similarity (excluding self)
        top_indices = np.argpartition(-cosine_sim[i], n_neighbors + 1)[1:n_neighbors + 1]

        # Compute angular distance weights
        for j in top_indices:
            cos_ij = max(min(cosine_sim[i, j], 1.0), -1.0)  # Numerical safety
            E[i, j] = np.arccos(-cos_ij)  # Your specified E_ij formula

    # Symmetrize the graph (undirected)
    E = np.maximum(E, E.T)

    # Step 2: Construct normalized transition matrix
    # 1. Compute row sums of E (W in the paper) -> D_ii = sum_j W_ij
    row_sums_E = E.sum(axis=1, keepdims=True)
    E = E / (row_sums_E + 1e-10)
    E = csr_matrix(E)
    D_rowsums = E.sum(axis=1).A1  # .A1 converts to 1D numpy array

    # 2. Compute D^{-1/2} (inverse square root of row sums)
    D_inv_sqrt = diags(1 / np.sqrt(D_rowsums + 1e-10))  # Add epsilon to avoid division by zero

    # 3. Construct T = D^{-1/2} E D^{-1/2} (E is W in the paper) # Ensure E is sparse
    T = D_inv_sqrt @ (E @ D_inv_sqrt)  # Parentheses optimize order

    # Initialize matrices for positive (S+) and negative (S-) seeds
    # Step 3: Prepare positive and negative seeds
    s_pos = np.zeros((n_words, 3))
    s_neg = np.zeros((n_words, 3))

    for word, (v, a, d) in seed_vad_scores.items():
        idx = word_to_idx[word]
        s_pos[idx] = [v, a, d]
        s_neg[idx] = [10 - v, 10 - a, 10 - d]  # Inverted around center (5)

    # Normalize seed vectors
    s_pos = sk_normalize(s_pos, axis=0, norm='l1')
    s_neg = sk_normalize(s_neg, axis=0, norm='l1')

    # Step 4: Run random walks with restart
    p_pos = np.ones((n_words, 3)) / n_words  # Uniform initialization
    p_neg = np.ones((n_words, 3)) / n_words

    # Run random walks for P+ and P-
    for i in range(max_iter):
        p_pos = beta * (T @ p_pos) + (1 - beta) * s_pos
        p_neg = beta * (T @ p_neg) + (1 - beta) * s_neg

    # Step 5: Compute final VAD scores (scaled to 1-9)
    epsilon = 1e-10  # Avoid division by zero
    P_final = p_pos / (p_pos + p_neg + epsilon)

    #saving and upscaling
    GOLD_words = get_gold_words(gold_standard_path)
    result_VAD_lexicon = [[word, P_final[i][0] * 8 + 1, P_final[i][1] * 8 + 1, P_final[i][2] * 8 + 1]
                          for i, word in enumerate(words) if word in GOLD_words]
    save_as_VAD(save_folder_name, vad_lexicon_name, embedding_name, result_VAD_lexicon)

