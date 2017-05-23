from emoint.featurizers.base_featurizers import SentimentLexiconFeaturizer
from emoint.featurizers.utils import bing_liu_lexicon_path

"""
Info: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
"""


class BingLiuFeaturizer(SentimentLexiconFeaturizer):
    """Bing Liu Sentiment Lexicon Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation

    def __init__(self, lexicons_path=bing_liu_lexicon_path):
        """Initialize BingLiu Sentiment Lexicon Featurizer
        :param lexicons_path path to lexicons file
        """
        self._id = 'BingLiu'
        self._lexicon_map = self.create_lexicon_mapping(lexicons_path)
        self._citation = 'Hu, Minqing, and Bing Liu. "Mining and summarizing customer reviews."' \
                         ' Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery' \
                         ' and data mining. ACM, 2004.'
