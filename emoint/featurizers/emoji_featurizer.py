from emoint.featurizers.base_featurizers import EmbeddingFeaturizer
from emoint.featurizers.utils import emoji_embedding_path

"""
Info: https://github.com/uclmr/emoji2vec
"""


class EmojiEmbeddingsFeaturizer(EmbeddingFeaturizer):
    """Emoji Embeddings Featurizer"""

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

    def __init__(self, embedding_path=emoji_embedding_path, dim=300):
        """Initialize Emoji Embeddings Featurizer
        :param embedding_path path to embeddings file
        """
        self._id = 'Emoji'
        self._dim = dim
        self._embedding_map = self.create_embedding_mapping(embedding_path)
        self._citation = 'Eisner, Ben, et al. "emoji2vec: Learning Emoji Representations from their Description."' \
                         ' arXiv preprint arXiv:1609.08359 (2016).'
