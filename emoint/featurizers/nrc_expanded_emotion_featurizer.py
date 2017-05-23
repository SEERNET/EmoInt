# coding=utf-8
from emoint.featurizers.base_featurizers import EmotionLexiconFeaturizer
from emoint.featurizers.utils import nrc_expanded_emotion_lexicon_path

"""
Info: http://www.cs.waikato.ac.nz/ml/sa/lex.html
"""


class NRCExpandedEmotionFeaturizer(EmotionLexiconFeaturizer):
    """NRC Expanded Emotion Lexicon Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation

    def __init__(self, lexicon_path=nrc_expanded_emotion_lexicon_path):
        """Initialize Bravo-Marquez NRC Expanded Emotion Lexicon featurizer
        :param lexicon_path path to unigram lexicons file
        """
        self._id = 'NRCExpandedEmotion'
        self._lexicon_map = self.create_lexicon_mapping(lexicon_path)
        self._citation = 'Bravo-Marquez, Felipe, et al. "Determining word–emotion associations from tweets by' \
                         ' multi-label classification." WI\'16. IEEE Computer Society, 2016.'
