from emoint.featurizers.base_featurizers import EmotionLexiconFeaturizer
from emoint.featurizers.utils import nrc_emotion_lexicon_path

"""
Info: http://saifmohammad.com/WebPages/lexicons.html
"""


class NRCEmotionFeaturizer(EmotionLexiconFeaturizer):
    """NRC Wordlevel Emotion Lexicon Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation

    def __init__(self, lexicon_path=nrc_emotion_lexicon_path):
        """Initialize Saif Mohammad NRC Wordlevel Emotion Lexicon featurizer
        :param lexicon_path path to unigram lexicons file
        """
        self._id = 'NRCEmotionWordlevel'
        self._lexicon_map = self.create_lexicon_mapping(lexicon_path)
        self._citation = 'Mohammad, Saif M., and Peter D. Turney. "Emotions evoked by common words and phrases:' \
                         ' Using Mechanical Turk to create an emotion lexicon." Proceedings of the NAACL HLT 2010' \
                         ' workshop on computational approaches to analysis and generation of emotion in text.' \
                         ' Association for Computational Linguistics, 2010.'
