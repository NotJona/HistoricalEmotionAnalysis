"""
This file implements PaRaSim
"""
from functions import load_seed_data, load_GOLD_data, cosine_similarity, save_as_VAD
import numpy as np

def paRaSim(embedding_name, lexicon_path, gold_standard_path, vad_lexicon_name, save_folder_name):
    """
    Uses the PaRaSim algorithm to induce historical VAD scores. Saves result under:
    'HistoricalVAD/PaRaSim/vad_lexicon_name _ model_name _ historicalVAD.tsv'
    :param embedding_name: str, name of the embedding
    :param lexicon_path: str, path to seed word lexicon
    :param gold_standard_path: str, path to gold standard lexicon
    :param vad_lexicon_name: str, name of the seed word lexicon (important for saving)
    :param save_folder_name: str, name of the saved folder
    """
    result_VAD_lexicon = []
    seed_data = load_seed_data(embedding_name, lexicon_path)
    seed_VAD = np.vstack([element[1] for element in seed_data])
    seed_vectors = np.vstack([element[2] for element in seed_data])
    seed_vector_norms = np.linalg.norm( seed_vectors, axis=1)
    GOLD_data = load_GOLD_data(embedding_name, gold_standard_path)

    for row in GOLD_data:
        val1 = np.dot(seed_vectors, row[1])/(np.linalg.norm(row[1])*seed_vector_norms)
        val2 = seed_VAD*val1[:, np.newaxis]
        vad_score = np.sum(val2, axis=0) / np.sum(val1)
        result_VAD_lexicon.append([row[0], vad_score[0], vad_score[1], vad_score[2]])

    save_as_VAD(save_folder_name, vad_lexicon_name, embedding_name, result_VAD_lexicon)

