from emoint.featurizers.base_featurizers import EmbeddingFeaturizer
from emoint.featurizers.utils import edinburgh_embedding_path

"""
Info: http://www.cs.waikato.ac.nz/~fjb11/publications/wi2016a.pdf
"""


class EdinburghEmbeddingsFeaturizer(EmbeddingFeaturizer):
    """Edinburgh Embeddings Featurizer"""

    @property
    def dim(self):
        return self._dim

    @property
    def embedding_map(self):
        return self._embedding_map

    @property
    def id(self):
        return self._id

    @property
    def citation(self):
        return self._citation

    def __init__(self, embedding_path=edinburgh_embedding_path, dim=100, word_first=False, leave_head=False):
        """Initialize Edinburgh Embeddings Featurizer
        :param embedding_path path to embeddings file
        """
        self._id = 'Edinburgh'
        self._dim = dim
        self._embedding_map = self.create_embedding_mapping(embedding_path, word_first=word_first, leave_head=leave_head)
        self._citation = 'Bravo-Marquez, Felipe, Eibe Frank, and Bernhard Pfahringer.' \
                         ' "From unlabelled tweets to twitter-specific opinion words."' \
                         ' Proceedings of the 38th International ACM SIGIR Conference on Research and Development in' \
                         ' Information Retrieval. ACM, 2015.'
