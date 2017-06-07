from abc import ABCMeta, abstractmethod
import gzip
from collections import defaultdict


class Featurizer(object):
    """Base class for all featurizers"""

    __metaclass__ = ABCMeta

    @property
    def id(self):
        """A unique identifier of the featurizer"""
        raise NotImplementedError("Implement method to return unique featurizer identifier")

    @property
    def features(self):
        """Method to return names of the features"""
        raise NotImplementedError("Implement method to return names of the features")

    @property
    def dim(self):
        """Method to return dimension of the feature space"""
        return len(self.features)

    @abstractmethod
    def featurize(self, tokens):
        """Method to create features from the tokens"""
        raise NotImplementedError("Implement method to featurize tokens")

    @property
    def citation(self):
        """Add citation to papers or methods being implemented. Give credit where credit is due."""
        raise NotImplementedError("Implement method to return citation of the paper or method being implemented")


class LexiconFeaturizer(Featurizer):
    """Featurizer where we map lexicons to features"""

    __metaclass__ = ABCMeta

    @property
    def lexicon_map(self):
        """Method to populate lexicon to features mapping"""
        raise NotImplementedError("Implement method to populate lexicon to features mapping")

    def create_lexicon_mapping(self, lexicon_path):
        """Method to create lexicon mapping"""


class EmbeddingFeaturizer(Featurizer):
    """Featurizer where we map lexicons to word embeddings"""

    __metaclass__ = ABCMeta

    @property
    def dim(self):
        """Dimension of word embeddings"""
        raise NotImplementedError("Implement method to return dimension of word embeddings")

    @property
    def features(self):
        return [self.id + '-' + str(x) for x in range(self.dim)]

    @property
    def embedding_map(self):
        """Method to populate lexicon to features mapping"""
        raise NotImplementedError("Implement method to populate lexicon to features mapping")

    def featurize(self, tokens):
        """Averaging word embeddings"""
        sum_vec = [0.0] * self.dim
        for token in tokens:
            if token in self.embedding_map:
                sum_vec = [a + b for a, b in zip(sum_vec, self.embedding_map[token])]
        denom = len(tokens)
        sum_vec = [num / denom for num in sum_vec]
        return sum_vec

    def create_embedding_mapping(self, lexicon_path):
        """Creates a map from words to word embeddings
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(lexicon_path) as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.split('\t')
                # assert (self.dim == len(splits) - 1)
                lexicon_map[splits[-1]] = [float(num) for num in splits[:-1]]
        return lexicon_map


class SentimentLexiconFeaturizer(LexiconFeaturizer):
    """Sentiment Featurizer is where we have mappings from lexicons to Sentiment (Positive, Negative)"""

    __metaclass__ = ABCMeta

    @property
    def features(self):
        return [self.id + '-' + x for x in ['positive', 'negative']]

    def create_lexicon_mapping(self, lexicon_path):
        """Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(lexicon_path) as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.split('\t')
                lexicon_map[splits[0]] = splits[1]
        return lexicon_map

    def featurize(self, tokens):
        """This function returns count of positive and negative tokens
        :param tokens list of tokens
        """
        positive_count, negative_count = 0.0, 0.0
        for token in tokens:
            if token in self.lexicon_map:
                if self.lexicon_map[token] == 'positive':
                    positive_count += 1
                else:
                    negative_count += 1
        return [positive_count, negative_count]


class SentimentIntensityLexiconFeaturizer(LexiconFeaturizer):
    """Sentiment Intensity Featurizer is where we have mappings from lexicons to Intensity (Positive, Negative)"""

    __metaclass__ = ABCMeta

    @property
    def features(self):
        return [self.id + '-' + x for x in ['PositiveScore', 'NegativeScore']]

    def create_lexicon_mapping(self, lexicon_path):
        """Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(lexicon_path) as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.split('\t')
                lexicon_map[splits[0]] = float(splits[1])
        return lexicon_map

    def featurize(self, tokens):
        """This function returns sum of intensities of positive and negative tokens
        :param tokens list of tokens
        """
        positive_score, negative_score = 0.0, 0.0
        for token in tokens:
            if token in self.lexicon_map:
                if self.lexicon_map[token] >= 0:
                    positive_score += self.lexicon_map[token]
                else:
                    negative_score += self.lexicon_map[token]
        return [positive_score, negative_score]


class EmotionLexiconFeaturizer(LexiconFeaturizer):
    """Sentiment Intensity Featurizer is where we have mappings from lexicons to Intensity (Positive, Negative)"""

    __metaclass__ = ABCMeta

    @staticmethod
    def emotions():
        return ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                'negative', 'positive', 'sadness', 'surprise', 'trust']

    def get_feature_name(self, emotion):
        return self.id + "-" + emotion.capitalize()

    @property
    def features(self):
        return [self.get_feature_name(emotion) for emotion in self.emotions()]

    def create_lexicon_mapping(self, lexicon_path):
        """Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(lexicon_path) as f:
            lines = f.read().splitlines()
            lexicon_map = defaultdict(list)

            headers = lines[0].split('\t')[1:]
            for l in lines[1:]:
                splits = l.split('\t')
                lexicon_map[splits[0]] = [float(num) for num in splits[1:]]

        return lexicon_map

    def featurize(self, tokens):
        """This function returns score of tokens belonging to different emotions
        :param tokens list of tokens
        """
        sum_vec = [0.0] * len(self.features)
        for token in tokens:
            if token in self.lexicon_map:
                sum_vec = [a + b for a, b in zip(sum_vec, self.lexicon_map[token])]
        return sum_vec
