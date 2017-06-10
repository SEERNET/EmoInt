from emoint.featurizers.base_featurizers import SentimentIntensityLexiconFeaturizer
from emoint.featurizers.utils import senti_wordnet_lexicon_path
import gzip
from collections import defaultdict

"""
Info: http://sentiwordnet.isti.cnr.it/
"""


class SentiWordNetFeaturizer(SentimentIntensityLexiconFeaturizer):
    """Senti WordNet Featurizer"""

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def id(self):
        return self._id

    @property
    def citation(self):
        return self._citation

    def create_lexicon_mapping(self, lexicon_path):
        """Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(lexicon_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = defaultdict(float)
            for l in lines:
                l = l.decode('utf-8')
                if l.strip().startswith('#'):
                    continue
                splits = l.split('\t')
                # positive score - negative score
                score = float(splits[2]) - float(splits[3])
                words = splits[4].split(" ")
                # iterate through all words
                for word in words:
                    word, rank = word.split('#')
                    # scale scores according to rank
                    # more popular => less rank => high weight
                    lexicon_map[word] += (score / float(rank))

        return lexicon_map

    def __init__(self, lexicon_path=senti_wordnet_lexicon_path):
        """Initialize Senti WordNet featurizer
        :param lexicon_path path to unigram lexicons file
        """
        self._id = 'SentiWordNet'
        self._lexicon_map = self.create_lexicon_mapping(lexicon_path)
        self._citation = 'Baccianella, Stefano, Andrea Esuli, and Fabrizio Sebastiani. "SentiWordNet 3.0: An Enhanced' \
                         ' Lexical Resource for Sentiment Analysis and Opinion Mining." LREC. Vol. 10. 2010.'
