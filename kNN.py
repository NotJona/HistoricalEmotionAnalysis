"""
This file implements kkN
"""
from functions import load_seed_data, load_GOLD_data, cosine_similarity, save_as_VAD
import numpy as np

def kNN(embedding_name, lexicon_path, gold_standard_path, vad_lexicon_name, save_folder_name, k=16):
    """
    Uses the kNN algorithm to induce historical VAD scores. Saves result under:
    'HistoricalVAD/kNN/vad_lexicon_name _ model_name _ historicalVAD.tsv'
    :param embedding_name: str, name of the embedding
    :param lexicon_path: str, path to seed word lexicon
    :param gold_standard_path: str, path to gold standard lexicon
    :param vad_lexicon_name: str, name of the seed word lexicon (important for saving)
    :param save_folder_name: str, name of the saved folder
    :param k: int, number of neighbors to use (following Hellrich et al.)
    """
    result_VAD_lexicon = []
    seed_data = load_seed_data(embedding_name, lexicon_path)
    GOLD_data = load_GOLD_data(embedding_name, gold_standard_path)
    for row in GOLD_data:
        similarity_VAD_index = dict()
        for element in seed_data:
            sim = cosine_similarity(row[1], element[2])
            similarity_VAD_index[sim] = (element[0], element[1])
        top_keys = sorted(similarity_VAD_index.keys(), reverse=True)[:k]
        top_values = [similarity_VAD_index[key] for key in top_keys]
        vad_score= np.sum([element[1] for element in top_values], axis= 0)/k
        result_VAD_lexicon.append([row[0],vad_score[0], vad_score[1], vad_score[2]])

    save_as_VAD(save_folder_name, vad_lexicon_name, embedding_name, result_VAD_lexicon)

