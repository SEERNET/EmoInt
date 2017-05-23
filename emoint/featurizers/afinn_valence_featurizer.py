# coding=utf-8
from emoint.featurizers.base_featurizers import SentimentIntensityLexiconFeaturizer
from emoint.featurizers.utils import get_bigrams
from emoint.featurizers.utils import afinn_lexicon_path, afinn_emoticon_path

"""
Info: http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
"""


class AFINNValenceFeaturizer(SentimentIntensityLexiconFeaturizer):
    """AFINN Valence Lexicons Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation

    def __init__(self, lexicons_path=afinn_lexicon_path, emoticon_path=afinn_emoticon_path, bigram=True):
        """Initialize Finn Årup Nielsen Valence Lexicon Featurizer
        :param lexicons_path path to lexicons file
        :param emoticon_path path to emoticons file
        :param bigram use bigram lexicons or not (default: True)
        """
        self._id = 'AFINN'
        self._lexicon_map = dict(self.create_lexicon_mapping(lexicons_path).items() +
                                 self.create_lexicon_mapping(emoticon_path).items())
        self._citation = 'Nielsen, Finn Årup. "A new ANEW: Evaluation of a word list for sentiment analysis in' \
                         ' microblogs." arXiv preprint arXiv:1103.2903 (2011).'
        self.bigram = bigram

    def featurize(self, tokens):
        """Featurize tokens using AFINN Valence Lexicons
        :param tokens tokens to featurize
        """
        unigrams = tokens
        if self.bigram:
            bigrams = get_bigrams(tokens)
            return [x + y for x, y in zip(super(AFINNValenceFeaturizer, self).featurize(unigrams),
                                          super(AFINNValenceFeaturizer, self).featurize(bigrams))]
        else:
            return super(AFINNValenceFeaturizer, self).featurize(unigrams)
