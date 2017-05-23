from emoint.featurizers.base_featurizers import SentimentIntensityLexiconFeaturizer
from emoint.featurizers.utils import get_bigrams, sentiment140_unigram_lexicon_path, sentiment140_bigram_lexicon_path

"""
Info: http://saifmohammad.com/WebPages/lexicons.html
"""


class Sentiment140Featurizer(SentimentIntensityLexiconFeaturizer):
    """NRC Hashtag Sentiment Lexicon featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation

    def __init__(
            self,
            unigram_lexicons_path=sentiment140_unigram_lexicon_path,
            bigram_lexicons_path=sentiment140_bigram_lexicon_path,
            bigram=True
    ):
        """Initialize Saif Mohammad Sentiment140 Lexicon featurizer
        :param unigram_lexicons_path path to unigram lexicons file
        :param bigram_lexicons_path path to bigram lexicons file
        :param bigram use bigram lexicons or not (default: True)
        """
        self._id = 'Sentiment140'
        self._lexicon_map = dict(
            self.create_lexicon_mapping(unigram_lexicons_path).items() +
            self.create_lexicon_mapping(bigram_lexicons_path).items()
        )
        self._citation = 'Mohammad, Saif M., Svetlana Kiritchenko, and Xiaodan Zhu. ' \
                         '"NRC-Canada: Building the state-of-the-art in sentiment analysis of tweets."' \
                         ' arXiv preprint arXiv:1308.6242 (2013).'
        self.bigram = bigram

    def featurize(self, tokens):
        """Featurize tokens using Saif Mohammad Sentiment140 Lexicon featurizer
        :param tokens tokens to featurize
        """
        unigrams = tokens
        if self.bigram:
            bigrams = get_bigrams(tokens)
            return [x + y for x, y in zip(super(Sentiment140Featurizer, self).featurize(unigrams),
                                          super(Sentiment140Featurizer, self).featurize(bigrams))]
        else:
            super(Sentiment140Featurizer, self).featurize(unigrams)
