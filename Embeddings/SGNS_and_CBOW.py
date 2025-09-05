"""
This file contains the code for training the SGNS and CBOW word embeddings.
Takes a corpus and returns models.

Follows the documentation provided by the gensim package:
https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
https://radimrehurek.com/gensim/models/word2vec.html

Hyperparameters taken from Hellrich et al. who follow Hamilton 2016b who follow Levy 2015

NOTE: If negative samples = 5, PPMI might need to be shifted by log(5) for better comparison!!!

This file was written with the help of generative AI.
"""
from gensim.models import Word2Vec
import json
from pathlib import Path

#Global Variables
sgns_matrix_path = Path('Embeddings/SGNS.model')
cbow_matrix_path = Path('Embeddings/CBOW.model')

def compute_sgns_matrix(corpus_path):

    """MODELPARAMETERS for SGNS"""
    vector_size = 300  # Following Hellrich et al. (gensim default = 100)
    alpha = 0.025  # learing rate (gensim default)
    window = 4  # context window following Hellrich et al. (gensim default = 5)
    min_count = 10  # Ignores all words with total frequency lower than this. Following Hellrich et al.(gensim default = 5)
    sample = 0.00001  # threshold for configuring which higher-frequency words are randomly downsampled. Following Hamilton (gensim default 0.001)
    min_alpha = 0.0001  # minimal value of the learning rate (gensim default)
    sg = 1  # Training algorithm: 1 for skip-gram; otherwise CBOW. (gensim default = 0)
    hs = 0  # If 1, hierarchical softmax will be used for model training, if 0 not (gensim default)
    negative = 5  # number of negative samples (gensim default, but follows Hamilton)
    ns_exponent = 0.75  # context distribution smoothing parameter (gensim default, but follows Levy)
    epochs = 8  # (5 is gensim default)

    if sgns_matrix_path.exists():
        print('Embedding already computed')
    else:
        print('Computing SGNS matrix')
        with open(corpus_path, encoding='utf-8') as f:
            corpus = json.load(f)
        model = Word2Vec(sentences=corpus, vector_size=vector_size, alpha=alpha, window=window, min_count=min_count,
                         sample=sample, min_alpha=min_alpha, sg=sg, hs=hs, negative=negative,
                         ns_exponent=ns_exponent, epochs=epochs)
        model.save(str(sgns_matrix_path))
        print('SGNS model saved')


def compute_cbow_matrix(corpus_path):

    """MODELPARAMETERS for CBOW"""
    vector_size = 300  # Following Hellrich et al. (gensim default = 100)
    alpha = 0.025  # learing rate (gensim default)
    window = 4  # context window following Hellrich et al. (gensim default = 5)
    min_count = 10  # Ignores all words with total frequency lower than this. Following Hellrich et al.(gensim default = 5)
    sample = 0.00001  # threshold for configuring which higher-frequency words are randomly downsampled. Following Hamilton (gensim default 0.001)
    min_alpha = 0.0001  # minimal value of the learning rate (gensim default)
    cbow = 0  # Training algorithm: 1 for skip-gram; otherwise CBOW. (gensim default = 0)
    hs = 0  # If 1, hierarchical softmax will be used for model training, if 0 not (gensim default)
    negative = 5  # number of negative samples (gensim default, but follows Hamilton)
    ns_exponent = 0.75  # context distribution smoothing parameter (gensim default, but follows Levy)
    epochs = 20  # (gensim default)

    if cbow_matrix_path.exists():
        print('Embedding already computed')
    else:
        print('Computing CBOW matrix')
        with open(corpus_path, encoding='utf-8') as f:
            corpus = json.load(f)
        model = Word2Vec(sentences=corpus, vector_size=vector_size, alpha=alpha, window=window, min_count=min_count,
                             sample=sample, min_alpha=min_alpha, sg=cbow, hs=hs, negative=negative,
                             ns_exponent=ns_exponent, epochs=epochs)
        model.save(str(cbow_matrix_path))
        print('CBOW model saved')

