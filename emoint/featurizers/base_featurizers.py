from abc import ABCMeta, abstractmethod
import gzip
from collections import defaultdict
import numpy as np


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
    def featurize(self, text, tokenizer):
        """Method to create features from text"""
        raise NotImplementedError("Implement method to featurize text")

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

    def featurize(self, text, tokenizer):
        """Averaging word embeddings
        """
        sum_vec = np.zeros(shape=(self.dim,))
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            if token in self.embedding_map:
                sum_vec = sum_vec + self.embedding_map[token]
        denom = len(tokens)
        sum_vec = sum_vec / denom
        return sum_vec

    @staticmethod
    def create_embedding_mapping(lexicon_path):
        """Creates a map from words to word embeddings
        :param lexicon_path path of lexicon file (in gzip format)
        """
        import numpy as np
        print("Started createing")
        i = 0
        with open(lexicon_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                if i%1000000 == 0:
                    print(i)
                i = i+1
                splits = l.split(' ')
                lexicon_map[splits[0]] = np.asarray(splits[1:], dtype='float32')
        print("Started createing")
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
        with gzip.open(lexicon_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = splits[1]
        return lexicon_map

    def featurize(self, text, tokenizer):
        """This function returns count of positive and negative tokens
        :param text: text to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        positive_count, negative_count = 0.0, 0.0
        tokens = tokenizer.tokenize(text)
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
        with gzip.open(lexicon_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = float(splits[1])
        return lexicon_map

    def featurize(self, text, tokenizer):
        """This function returns sum of intensities of positive and negative tokens
        :param text: text to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        positive_score, negative_score = 0.0, 0.0
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            if token in self.lexicon_map:
                if self.lexicon_map[token] >= 0:
                    positive_score += self.lexicon_map[token]
                else:
                    negative_score += self.lexicon_map[token]
        return [positive_score, negative_score]

    def featurize_tokens(self, tokens):
        """This function returns sum of intensities of positive and negative tokens
        :param tokens tokens to featurize
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

    def get_missing(self, text, tokenizer):
        tokens = tokenizer.tokenize(text)
        tc, mc = 0.0, 0.0
        for token in tokens:
            tc += 1
            if not token in self.lexicon_map:
                mc += 1
        if tc == 0:
            return 1.0
        else:
            # print("Total: {}, Missing: {}".format(tc, mc * 1.0 / tc * 1.0))
            return mc * 1.0 / tc * 1.0

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
        with gzip.open(lexicon_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = defaultdict(list)

            for l in lines[1:]:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = [float(num) for num in splits[1:]]

        return lexicon_map

    def featurize(self, text, tokenizer):
        """This function returns score of tokens belonging to different emotions
        :param text: text to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        sum_vec = [0.0] * len(self.features)
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            if token in self.lexicon_map:
                sum_vec = [a + b for a, b in zip(sum_vec, self.lexicon_map[token])]
        return sum_vec

