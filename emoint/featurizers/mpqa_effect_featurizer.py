from emoint.featurizers.base_featurizers import SentimentLexiconFeaturizer
from emoint.featurizers.utils import mpqa_lexicon_path

"""
Info: http://mpqa.cs.pitt.edu/lexicons/effect_lexicon/
"""


class MPQAEffectFeaturizer(SentimentLexiconFeaturizer):
    """MPQA Effect Lexicon Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation

    def __init__(self, lexicons_path=mpqa_lexicon_path):
        """Initialize MPQA Effect Lexicon Featurizer
        :param lexicons_path path to lexicons file
        """
        self._id = 'MPQA'
        self._lexicon_map = self.create_lexicon_mapping(lexicons_path)
        self._citation = 'Choi, Yoonjung, and Janyce Wiebe. "+/-EffectWordNet:' \
                         ' Sense-level Lexicon Acquisition for Opinion Inference." EMNLP. 2014.'
