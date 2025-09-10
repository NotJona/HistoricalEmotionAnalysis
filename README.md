# HistoricalEmotionAnalysis

This is the repository of my master's thesis "Computational Linguistics Approaches to Historical Emotion Analysis: Evaluating Word Embeddings, Induction Algorithms, and Lexicon Choice."

# Abstract

This master’s thesis investigates potential extensions of the model for historical emotion analysis proposed by Hellrich et al. (2019). For this purpose, the model – based on the combination of historical word embedding, a VAD lexicon, and an induction algorithm – is extended by three word embeddings (PPMI, CBOW, FastText), an induction algorithm (linear regression), and a VAD lexicon (NRC-VAD). In addition, the influence of lexicon size is examined. Model performance can be successfully improved: the best model combines SGNS embedding, linear regression as induction algorithm, and the largest version of the Warriner VAD lexicon used in the original model. However, the results indicate that a stronger contemporary influence has a positive effect on historical emotion analysis. Possible causes for this contradictory outcome are discussed.

# Notes

Note that (1) the following data could not be included due to size constraints:
- for Section 4.1 (Preprocessing): the preprocessed corpora, the trained embeddings
- for Section 4.3 (Investigating Lexicon Size): the randomly generated seed word lexica, the predicted historical vad lexica

Note that (2) the COHA could not be included as it is not open data. I did include an 'Example Corpus' and example preprocessed data (preprocessed_corpus.json and preprocessed_filtered_corpus.json are NOT the corpus files I used, they are here as example data!!), so the code can be tested. But, of course, one cannot recreate my findings with these data alone.

Note that (3), if you decide to retrain the embeddings yourself, results will vary slightly due to the non-deterministic nature of SGNS, CBOW, and FastText. While some outcomes in 4.2 and 4.3 may change, my central findings of 4.4 (discussed in 5.) still hold true.  
