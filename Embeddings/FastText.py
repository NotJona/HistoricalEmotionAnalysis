"""
This file contains code for training a FastText model.
It takes a corpus and returns a model

This file was written with the help of generative AI.
"""
from gensim.models import FastText
import json
from pathlib import Path

#Global Variables
fasttext_matrix_path = Path('Embeddings/fasttext_model.bin')

"""MODELPARAMETERS for FastText"""
vector_size = 300   # Following Hellrich et al. (gensim default = 100)
window = 4          # context window following Hellrich et al. (gensim default = 5)
min_count = 10      # Ignores all words with total frequency lower than this. Following Hellrich et al.(gensim default = 5)
workers = 4         # Use 4 CPU cores
sample = 0.00001    # threshold for configuring which higher-frequency words are randomly downsampled. Following Hamilton (gensim default 0.001)
alpha = 0.025       # Initial learning rate
min_alpha = 0.0001  # minimal value of the learning rate (gensim default)
sg = 1              # Training algorithm: 1 for skip-gram; otherwise CBOW. (gensim default = 0)
hs = 0              # If 1, hierarchical softmax will be used for model training, if 0 not (gensim default)
negative = 5        # number of negative samples (gensim default, but follows Hamilton)
ns_exponent = 0.75  # context distribution smoothing parameter (gensim default, but follows Levy)
min_n = 3           # Min subword n-gram length
max_n = 6           # Max subword n-gram length
epochs = 35         # Training iterations (gensim default = 10)

def compute_fasttext_matrix(corpus_path):
    if fasttext_matrix_path.exists():
        print('Embedding already computed')
    else:
        print('Computing FastText Matrix ...')
        with open(corpus_path, encoding='utf-8') as f:
            corpus = json.load(f)
        model = FastText(
            sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
            sample=sample, alpha=alpha, min_alpha=min_alpha, sg=sg, hs=hs, negative=negative, ns_exponent=ns_exponent,
            min_n=min_n, max_n=max_n, epochs=epochs
        )
        model.save(str(fasttext_matrix_path))
        print('FastText matrix saved')

