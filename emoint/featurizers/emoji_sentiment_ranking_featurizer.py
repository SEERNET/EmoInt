# coding=utf-8
from emoint.featurizers.base_featurizers import Featurizer
from emoint.featurizers.utils import emoji_sentiment_ranking_path

import csv

"""
Info: http://kt.ijs.si/data/Emoji_sentiment_ranking/about.html
"""


class EmojiSentimentRanking(Featurizer):
    """Emoji Sentiment Ranking Featurizer"""

    @property
    def features(self):
        return [self.id + '_' + x for x in ['Negative', 'Neutral', 'Positive']]

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def citation(self):
        return self._citation


    @staticmethod
    def create_lexicon_mapping(lexicons_path):
        rows = csv.DictReader(open(lexicons_path), )
        emoji_map = {}
        for row in rows:
            emoji_map[row['Emoji'].decode('utf-8')] = [
                float(row['Negative'])/float(row['Occurrences']),
                float(row['Positive'])/float(row['Occurrences']),
                float(row['Neutral'])/float(row['Occurrences'])
            ]
        return emoji_map

    def __init__(self, lexicons_path=emoji_sentiment_ranking_path):
        """Initialize Novak Emoji Sentiment Ranking Featurizer
        :param lexicons_path path to lexicons file
        """
        self._id = 'Emoji_Sentiment_Ranking'
        self._lexicon_map = self.create_lexicon_mapping(lexicons_path)
        self._citation = 'Kralj Novak, Petra; Smailović, Jasmina; Sluban, Borut; et al., 2015,' \
                         ' Emoji Sentiment Ranking 1.0, Slovenian language resource repository CLARIN.SI,' \
                         ' http://hdl.handle.net/11356/1048.'

    def featurize(self, text, tokenizer):
        """Featurize tokens using Novak Emoji Sentiment Ranking Lexicons
        :param text: text to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        tokens = tokenizer.tokenize(text)
        sum_vec = [0.0] * len(self.features)
        for token in tokens:
            if token in self.lexicon_map:
                sum_vec = [a + b for a, b in zip(sum_vec, self.lexicon_map[token])]
        return sum_vec


