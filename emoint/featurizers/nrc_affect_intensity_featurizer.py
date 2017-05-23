from emoint.featurizers.base_featurizers import EmotionLexiconFeaturizer
from emoint.featurizers.utils import nrc_affect_intensity_lexicon_path

"""
Info: http://saifmohammad.com/WebPages/AffectIntensity.htm
"""


class NRCAffectIntensityFeaturizer(EmotionLexiconFeaturizer):
    """NRC Affect Intensity Lexicon Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation

    def __init__(self, lexicon_path=nrc_affect_intensity_lexicon_path):
        """Initialize Saif Mohammad NRC Affect Intensity Featurizer
        :param lexicon_path path to lexicons file
        """
        self._id = 'NRCAffectIntensity'
        self._lexicon_map = self.create_lexicon_mapping(lexicon_path)
        self._citation = 'Mohammad, Saif M. "Word Affect Intensities." arXiv preprint arXiv:1704.08798 (2017).'
