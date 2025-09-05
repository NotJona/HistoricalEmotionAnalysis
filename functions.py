"""
Important functions used for the algorithms
"""

from gensim.models import FastText
from gensim.models import Word2Vec
import numpy as np
from scipy import sparse
from pathlib import  Path

word_index_dict = {}
index_word_dict = {}
with open("Embeddings/AAA_word_list.txt", "r") as file:
    for index, line in enumerate(file):
        word_index_dict[line.strip()] = index
        index_word_dict[index] = line.strip()

def get_gold_words(gold_standard_path):
    """
    extracts words from gold standard file.
    :param gold_standard_path: str, path to gold standard file.
    :return: list, list of gold standard words.
    """
    GOLD_words = []
    with open(gold_standard_path) as file:
        for line in file:
            line = line.split('\t')
            GOLD_words.append(line[0])
    return GOLD_words

def cosine_similarity(vec1, vec2):
    return np.dot(vec1.T, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_model(embedding_name):
    """
    loads the different models. Takes 'SGNS', 'CBOW', 'FastText', 'SVD_PPMI' or 'PPMI as input'
    :param embedding_name: str, name of embedding.
    :return: loaded embedding.
    """
    if embedding_name == 'SGNS':
        model = Word2Vec.load('Embeddings/SGNS.model')
    elif embedding_name == 'CBOW':
        model = Word2Vec.load('Embeddings/CBOW.model')
    elif embedding_name == 'FastText':
        model = FastText.load('Embeddings/fasttext_model.bin')
    elif embedding_name == 'SVD_PPMI':
        model = np.loadtxt('Embeddings/svd_ppmi_matrix.csv', delimiter=',', skiprows=1)
    elif embedding_name == "PPMI":
        model = sparse.load_npz('Embeddings/ppmi_matrix.npz').toarray()
    return model

def load_VAD_seed_lexion(lexicon_path):
    """
    loads a vad lexicon, i.e., its words and vad scores.
    :param lexicon_path: str, path to lexicon.
    :return: dict, returns dictionary of the form {word: vad_score}
    """
    lexicon = dict()
    with open(lexicon_path) as f:
        for l in f:
            l = l.split('\t')
            lexicon[l[0]] = np.array([float(l[1]), float(l[2]), float(l[3])])
    return lexicon

def load_seed_data(embedding_name, lexicon_path):
    """
    loads seed data, i.e., for a word it loads its vad score and the embedding vector (of the selected embedding).
    :param embedding_name: str, name of embedding.
    :param lexicon_path: str, path to lexicon.
    :return: list of tuples of the form [(word, VAD-score, word_vector), (word, VAD-score, word_vector), ...]
    """
    seed_data = []
    seed_lexicon = load_VAD_seed_lexion(lexicon_path)
    model = load_model(embedding_name)
    if embedding_name in ['SGNS', 'CBOW', 'FastText']:
        for word in seed_lexicon.keys():
            word_vector = model.wv[word]
            seed_data.append((word, seed_lexicon[word], word_vector))
    if embedding_name in ['SVD_PPMI', 'PPMI']:
        for word in seed_lexicon.keys():
            word_vector = model[word_index_dict[word]]
            seed_data.append((word, seed_lexicon[word], word_vector))
    return seed_data

def load_GOLD_data(embedding_name, gold_standard_path):
    """
    For each word it loads its embedding vector (of the selected embedding).
    :param embedding_name: str, name of embedding.
    :param gold_standard_path: str, path to gold standard file.
    :return: list of lists of the form [[word, word_vector], [word, word_vector], ...]
    """
    GOLD_data = []
    GOLD_words = get_gold_words(gold_standard_path)
    model = load_model(embedding_name)
    if embedding_name in ['SGNS', 'CBOW', 'FastText']:
        for word in GOLD_words:
            word_vector = model.wv[word]
            GOLD_data.append([word, word_vector])
    if embedding_name in ['SVD_PPMI', 'PPMI']:
        for word in GOLD_words:
            word_vector = model[word_index_dict[word]]
            GOLD_data.append([word, word_vector])
    return GOLD_data

def save_as_VAD(save_folder_name, vad_lexicon_name, embedding_name, list_word_VAD):
    """
    Saves a (predicted) vad lexicon as
    'HistoricalVAD/save_folder_name/vad_lexicon_name _ embedding_name _ historicalVAD.tsv'
    :param save_folder_name: str, path to save folder.
    :param vad_lexicon_name: str, path to vad lexicon.
    :param embedding_name: str, name of embedding.
    :param list_word_VAD: list of lists, of the form:
    [[word, Valence score, Arousal score, Dominance score], [word, Valence score, ...], ...]
    """
    save_dir = Path('HistoricalVAD/' + save_folder_name + '/')
    save_dir.mkdir(parents=True, exist_ok=True)

    save_file = save_dir / f'{vad_lexicon_name}_{embedding_name}_historicalVAD.tsv'

    with open(save_file, mode='w', encoding='utf-8') as f:
        for row in list_word_VAD:
            line = '\t'.join(map(str, row))
            f.write(line + '\n')
