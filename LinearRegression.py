"""
This file implements LinearRegression

Written with the help of GenAI
"""
import numpy as np
from sklearn.linear_model import Ridge
from functions import load_seed_data, load_GOLD_data, save_as_VAD

def linear_regression(embedding_name, lexicon_path, gold_standard_path, vad_lexicon_name, save_folder_name):
    """
    Uses linear regression (specifically Ridge regression) to induce historical VAD scores. Saves result under:
    'HistoricalVAD/LinearRegression/vad_lexicon_name _ model_name _ historicalVAD.tsv'
    :param embedding_name: str, name of the embedding
    :param lexicon_path: str, path to seed word lexicon
    :param gold_standard_path: str, path to gold standard lexicon
    :param vad_lexicon_name: str, name of the seed word lexicon (important for saving)
    :param save_folder_name: str, name of the saved folder
    """
    result_VAD_lexicon = []
    seed_data = load_seed_data(embedding_name, lexicon_path)
    GOLD_data = load_GOLD_data(embedding_name, gold_standard_path)
    X = []
    for element in seed_data:
        X.append(element[2])
    X = np.array(X)
    Y = []
    for element in seed_data:
        Y.append(element[1])
    Y = np.array(Y)

    # Ridge regression model (default alpha=1.0)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, Y)

    for row in GOLD_data:
        predicted_vad = ridge_model.predict([row[1]])
        result_VAD_lexicon.append([row[0], predicted_vad[0][0], predicted_vad[0][1], predicted_vad[0][2]])

    save_as_VAD(save_folder_name, vad_lexicon_name, embedding_name, result_VAD_lexicon)



