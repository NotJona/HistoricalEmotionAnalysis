"""
This file contains the 'Model' class

This file was written with the help of AI
"""
from pathlib import Path
from preprocessing import preprocess_coha, filter_coha
from correlation import calculate_correlation

class Model:
    ALLOWED_EMBEDDINGS = ['SGNS', 'CBOW', 'FastText', 'SVD_PPMI', 'PPMI']
    ALLOWED_INDUCTION_ALGORITHMS = ['kNN', 'PaRaSim', 'RandomWalk', 'LinearRegression']
    EMBEDDING_DICT = {'SGNS':Path('Embeddings/SGNS.model'),
                      'CBOW':Path('Embeddings/CBOW.model'),
                      'FastText': Path('Embeddings/fasttext_model.bin'),
                      'SVD_PPMI': Path('Embeddings/svd_ppmi_matrix.csv'),
                      'PPMI': Path('Embeddings/ppmi_matrix.npz')}

    def __init__(self, embedding_name, induction_algorithm_name, vad_lexicon_path, vad_lexicon_name,
                 gold_standard_path = Path('VADLexica/goldEN.vad'), corpus_path = Path('COHA/wlp_1830s_ksq') ):
        """
        :param embedding_name: str, name of the embedding to use. Can be 'PPMI', 'SVD_PPMI', 'SGNS', 'CBOW', or 'FastText'.
        :param induction_algorithm_name: str, name of the induction algorithm to be used. Can take the values 'kNN',
        'PaRaSim', 'RandomWalk', or 'LinearRegression'
        :param vad_lexicon_path: str or Path object, path of the seed word vad lexicon to be used
        :param vad_lexicon_name: str, name of the seed word vad lexicon to be used (important for saving)
        :param gold_standard_path: str or Path object, path of the gold standard vad lexicon to be used.
        :param corpus_path: str or Path object, path of unprocessed corpus.
        """
        self.embedding_name = embedding_name
        self.induction_algorithm_name = induction_algorithm_name

        """Preprocessing information"""
        self.corpus_path = corpus_path
        self.preprocessed_corpus_path = Path('preprocessed_corpus.json')
        self.preprocessed_filtered_corpus_path = Path('preprocessed_filtered_corpus.json')

        self.corpus_is_preprocessed = True if self.preprocessed_filtered_corpus_path.exists() else False
        self.corpus_number_of_documents = 0
        self.corpus_number_of_words = 0

        """Embedding information"""
        self.embedding_path = self.EMBEDDING_DICT[embedding_name]
        self.embedding_is_computed = True if self.embedding_path.exists() else False

        """Lexicon information"""
        self.vad_lexicon_path = vad_lexicon_path
        self.vad_lexicon_name = vad_lexicon_name
        self.gold_standard_path = gold_standard_path

    @property
    def embedding_name(self):
        return self._embedding_name

    @embedding_name.setter
    def embedding_name(self, value):
        if value not in self.ALLOWED_EMBEDDINGS:
            raise ValueError(f'Invalid: {value}, ´value has to be in {self.ALLOWED_EMBEDDINGS}')
        self._embedding_name = value

    @property
    def induction_algorithm_name(self):
        return self._induction_algorithm_name

    @induction_algorithm_name.setter
    def induction_algorithm_name(self, value):
        if value not in self.ALLOWED_INDUCTION_ALGORITHMS:
            raise ValueError(f'Invalid: {value}, value has to be in {self.ALLOWED_INDUCTION_ALGORITHMS}')
        self._induction_algorithm_name = value

    @property
    def vad_lexicon_path(self):
        return self._vad_lexicon_path

    @vad_lexicon_path.setter
    def vad_lexicon_path(self, value):
        path_obj = Path(value) if not isinstance(value, Path) else value
        if not path_obj.exists():
            raise FileNotFoundError(f'VAD lexicon path does not exist: {path_obj}')
        self._vad_lexicon_path = path_obj

    @property
    def gold_standard_path(self):
        return self._gold_standard_path

    @gold_standard_path.setter
    def gold_standard_path(self, value):
        path_obj = Path(value) if not isinstance(value, Path) else value
        if not path_obj.exists():
            raise FileNotFoundError(f'Gold Standard path does not exist: {path_obj}')
        self._gold_standard_path = path_obj

    @property
    def corpus_path(self):
        return self._corpus_path

    @corpus_path.setter
    def corpus_path(self, value):
        path_obj = Path(value) if not isinstance(value, Path) else value
        if not path_obj.exists():
            raise FileNotFoundError(f'Corpus path does not exist: {path_obj}')
        self._corpus_path = path_obj

    """METHODS"""

    def preprocess_corpus(self):
        """
        preprocesses the corpus and saves it under "preprocessed_corpus.json and preprocessed_filtered_corpus.json"
        """
        if self.preprocessed_filtered_corpus_path.exists():
            print('Preprocessed corpus already exists')
            self.corpus_is_preprocessed = True

        else:
            print('Preprocessing corpus ...')
            try:
                self.corpus_number_of_documents, self.corpus_number_of_words = preprocess_coha(self.corpus_path)
                if self.embedding_name in ['SVD_PPMI', 'PPMI']:
                    filter_coha(n=10)
                print('Corpus preprocessing completed successfully!')
                print(f'number of processed documents: {self.corpus_number_of_documents}.')
                print(f'size of preprocessed corpus: {self.corpus_number_of_words} words.')
                self.corpus_is_preprocessed = True

            except FileNotFoundError as e:
                print(f'File not found error during preprocessing: {e}')

            except Exception as e:
                print(f'Error during corpus preprocessing: {e}.')
                print(f'Please remember that your corpus must have the same structure as the Example Corpus.\n'
                      f'This includes the .txt files!')

    def train_embedding(self):
        """
        trains the embedding and saves it under 'Embeddings'
        """
        if not self.corpus_is_preprocessed:
            print('Corpus is not preprocessed!')
            print('Call method preprocess_corpus to preprocess the corpus.')
        else:
            if self.embedding_name == 'PPMI':
                from Embeddings.PPMI import compute_ppmi_matrix
                compute_ppmi_matrix(self.preprocessed_filtered_corpus_path)
                self.embedding_is_computed = True
            elif self.embedding_name == 'SVD_PPMI':
                from Embeddings.SVDPPMI import compute_svd_ppmi_matrix
                compute_svd_ppmi_matrix(self.preprocessed_filtered_corpus_path)
                self.embedding_is_computed = True
            elif self.embedding_name == 'FastText':
                from Embeddings.FastText import compute_fasttext_matrix
                compute_fasttext_matrix(self.preprocessed_corpus_path)
                self.embedding_is_computed = True
            elif self.embedding_name == 'SGNS':
                from Embeddings.SGNS_and_CBOW import compute_sgns_matrix, compute_cbow_matrix
                compute_sgns_matrix(self.preprocessed_corpus_path)
                self.embedding_is_computed = True
            else:
                from Embeddings.SGNS_and_CBOW import compute_sgns_matrix, compute_cbow_matrix
                compute_cbow_matrix(self.preprocessed_corpus_path)
                self.embedding_is_computed = True

    def induce_historical_vad_lexicon(self, save_folder_name = None):
        """
        induces a historical VAD lexicon and saves it under
        'HistoricalVAD/save_folder_name/vad_lexicon_name _ embedding_name _ historicalVAD.tsv'
        :param save_folder_name: str, the name of the folder to save the historical VAD lexicon
        """
        if save_folder_name is None:
            save_folder_name = self.induction_algorithm_name
        if not self.corpus_is_preprocessed:
            print('Corpus is not preprocessed!')
            print('Call method preprocess_corpus to preprocess the corpus.')
        elif not self.embedding_is_computed:
            print('Embedding is not computed!')
            print('Call method train_embedding.')
        else:
            if self.induction_algorithm_name == 'kNN':
                from kNN import kNN
                kNN(self.embedding_name, self.vad_lexicon_path, self.gold_standard_path,
                    self.vad_lexicon_name, save_folder_name)
            elif self.induction_algorithm_name == 'PaRaSim':
                from PaRaSim import paRaSim
                paRaSim(self.embedding_name, self.vad_lexicon_path, self.gold_standard_path,
                        self.vad_lexicon_name, save_folder_name)
            elif self.induction_algorithm_name == 'RandomWalk':
                from RandomWalk import random_walk
                random_walk(self.embedding_name, self.vad_lexicon_path, self.gold_standard_path,
                            self.vad_lexicon_name, save_folder_name)
            else:
                from LinearRegression import linear_regression
                linear_regression(self.embedding_name, self.vad_lexicon_path, self.gold_standard_path,
                                  self.vad_lexicon_name, save_folder_name)

    def compute_correlation(self, save_folder_name = None):
        """
        computes correlation between predicted historical VAD values and the gold standard ones
        :param save_folder_name:
        :return: list of lists, [r_V, p_V, r_A, p_a, r_D, p_D, r_mean]
        """
        if save_folder_name is None:
            save_folder_name = self.induction_algorithm_name
        path = Path(f'HistoricalVAD/{save_folder_name}/{self.vad_lexicon_name}_{self.embedding_name}_historicalVAD.tsv')

        if not path.exists():
            print('Historical VAD lexicon does not exist yet. Please run method "induce historical vad lexicon" first."')
        else:
            return calculate_correlation(self.gold_standard_path, path)



