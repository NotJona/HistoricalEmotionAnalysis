"""
This file follows Hellrich and al.'s approach that can be found under:
https://github.com/JULIELab/HistEmo/blob/master/latech-cfl-2019/corpus_preparation/convert-COHA.py

Hellrich et al. also removed all punctuation and converted to lowercase, hence I added the function clean_word in accordance with:
https://github.com/hellrich/hyperwords/blob/master/scripts/clean_corpus.sh

After preprocessing the resulting word count is 13,057,184. (or 12,873,074 if we filter out uncommon words)

This file was written with the help of generative AI.
"""

from pathlib import Path
import re
import json
from collections import Counter

def preprocess_coha(decade_file_path):
    """
    This function preprocesses one decade of the COHA (needs the files to have the identical form of the Example Corpus).
    It follows the procedure used by Hellrich et al.
    Preprocessed corpus is saved under 'preprocessed_corpus.json'
    :param decade_file_path: str, path to the decade.
    :return: list of lists, each sublist represents one document of the corpus.
    """
    corpus = []
    folder_path = Path(decade_file_path)

    for file_path in folder_path.glob("*.txt"):
        file = []
        with file_path.open("r", encoding="UTF-8") as f: #"ISO-8859-1"???
            for line in f:
                if not line.startswith("//"):
                    parts = line.strip().split('\t')
                    if linecheck(parts):
                        if parts[1] == '\x00':
                            it = parts[0]
                        else:
                            it = parts[1]
                        if it == "n't":
                            it = "not"
                        it = clean_word(it)
                        if it != "":
                            file.append(it)
        corpus.append(file)

    with open('preprocessed_corpus.json', "w", encoding="UTF-8") as f:
        json.dump(corpus, f)
    return len(corpus), sum([len(doc) for doc in corpus])


def filter_coha(n):
    """
    Assumes the existence of 'preprocessed_corpus.json'
    Filters out words that have a token frequency lower than n.
    (Gensim does this itself, but for PPMI and SVD I have to do it manually)
    Preprocessed and filtered corpus is saved under 'preprocessed_filtered_corpus.json'
    :param n: int, lower bound of the token frequency.
    """
    corpus = json.load(open("preprocessed_corpus.json", "r", encoding="UTF-8"))
    words = []
    filtered_corpus = []
    for doc in corpus:
        words.extend(doc)
    word_count = Counter(words)
    for doc in corpus:
        filtered_doc = []
        for word in doc:
            if word_count[word] >= n:
                filtered_doc.append(word)
        filtered_corpus.append(filtered_doc)

    with open('preprocessed_filtered_corpus.json', "w", encoding="UTF-8") as f:
        json.dump(filtered_corpus, f)


def linecheck(line):
    """
    Checks whether a line is a valid line.
    :param line: list
    :return: bool
    """
    result = (len(line) == 3 and line[0] != '@'
              and line[0] != 'q!' #this one i added myself it seems to be the marker of the end of a fictional text
              and not line[0].startswith('@@')
              and line[2] != 'null'
              and not (line[0] == '\x00' and line[1] == '\x00')
              and not line[0].startswith('&'))
    return result

def clean_word(word):
    """
    Cleans up a word following the Hyperwords toolkit.
    :param word: str, word to be cleaned.
    :return: str
    """
    # Convert to lowercase
    word = word.lower()
    # Replace sequences of non-alphanumeric characters around spaces with a single space
    word = re.sub(r'[^a-z0-9]*[\s]+[^a-z0-9]*', ' ', word)
    # Replace sequences of non-alphanumeric characters at the end of the string with a space
    word = re.sub(r'[^a-z0-9]*$', ' ', word)
    # Normalize multiple spaces to a single space
    word = re.sub(r'\s+', ' ', word).strip()
    if len(word) == 2 and '/' in word: #filters out '/z' and '/q' #this one I added myself!
        word = ''
    return word

