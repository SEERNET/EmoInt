from emoint.featurizers.base_featurizers import EmotionLexiconFeaturizer
from emoint.featurizers.utils import nrc_hashtag_emotion_lexicon_path

"""
Info: http://www.saifmohammad.com/WebPages/lexicons.html
"""


class NRCHashtagEmotionFeaturizer(EmotionLexiconFeaturizer):
    """NRC Hashtag Emotion Lexicon Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation

    def __init__(self, lexicon_path=nrc_hashtag_emotion_lexicon_path):
        """Initialize Said Mohammad NRC Hashtag Emotion Lexicon featurizer
        :param lexicon_path path to unigram lexicons file
        """
        self._id = 'NRCHashtagEmotion'
        self._lexicon_map = self.create_lexicon_mapping(lexicon_path)
        self._citation = 'Mohammad, Saif M., and Svetlana Kiritchenko. ' \
                         '"Using hashtags to capture fine emotion categories from tweets."' \
                         ' Computational Intelligence 31.2 (2015): 301-326.'
