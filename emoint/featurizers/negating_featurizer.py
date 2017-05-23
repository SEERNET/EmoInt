from emoint.featurizers.base_featurizers import Featurizer
from emoint.featurizers.utils import negation_lexicon_path
import gzip
from collections import defaultdict


class NegationFeaturizer(Featurizer):
    """Negation Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        raise NotImplementedError("Not Applicable")

    @property
    def features(self):
        return [self.id + "-" + "count"]

    def featurize(self, tokens):
        count = 0
        for token in tokens:
            if token in self.lexicon_map:
                count += 1
        return [count]

    def create_lexicon_mapping(self, lexicon_path):
        """Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(lexicon_path) as f:
            lines = f.read().splitlines()
            lexicon_map = defaultdict(int)
            for l in lines:
                splits = l.split('\t')
                lexicon_map[splits[0]] += 1
        return lexicon_map

    def __init__(self, lexicons_path=negation_lexicon_path):
        """Initialize Negation Count Featurizer
        :param lexicons_path path to lexicons file
        """
        self._id = 'Negation'
        self._lexicon_map = self.create_lexicon_mapping(lexicons_path)
