# coding=utf-8
from unittest import TestCase

from emoint.featurizers.afinn_valence_featurizer import AFINNValenceFeaturizer
from emoint.featurizers.bing_liu_sentiment_featurizer import BingLiuFeaturizer
from emoint.featurizers.edinburgh_embeddings_featurizer import EdinburghEmbeddingsFeaturizer
from emoint.featurizers.emoji_featurizer import EmojiEmbeddingsFeaturizer
from emoint.featurizers.mpqa_effect_featurizer import MPQAEffectFeaturizer
from emoint.featurizers.negating_featurizer import NegationFeaturizer
from emoint.featurizers.nrc_affect_intensity_featurizer import NRCAffectIntensityFeaturizer
from emoint.featurizers.nrc_emotion_wordlevel_featurizer import NRCEmotionFeaturizer
from emoint.featurizers.nrc_expanded_emotion_featurizer import NRCExpandedEmotionFeaturizer
from emoint.featurizers.nrc_hashtag_emotion_featurizer import NRCHashtagEmotionFeaturizer
from emoint.featurizers.nrc_hashtag_sentiment_featurizer import NRCHashtagSentimentFeaturizer
from emoint.featurizers.senti_wordnet_featurizer import SentiWordNetFeaturizer
from emoint.featurizers.sentiment140_featurizer import Sentiment140Featurizer
from emoint.featurizers.sentistrength import SentiStrengthFeaturizer


class TestMPQAEffectFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = MPQAEffectFeaturizer()
        got = featurizer.featurize(['abandoned'])
        expected = [0, 1]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestBingLiuFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = BingLiuFeaturizer()
        got = featurizer.featurize(['a+'])
        expected = [1, 0]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestAFINNValenceFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = AFINNValenceFeaturizer()
        got = featurizer.featurize(['can\'t', 'stand', ':)'])
        expected = [2, -3]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestSentiment140LexiconFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = Sentiment140Featurizer()
        got = featurizer.featurize(['bad', 'sunday'])
        expected = [0.267, -1.297 + -4.999]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestNRCHashtagSentimentFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = NRCHashtagSentimentFeaturizer()
        got = featurizer.featurize(['bad', 'day'])
        expected = [0.831 + 0.395, -0.751]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestNRCEmotionFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = NRCEmotionFeaturizer()
        got = featurizer.featurize(['bad'])
        expected = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestNRCAffectIntensityFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = NRCAffectIntensityFeaturizer()
        got = featurizer.featurize(['bad'])
        expected = [0.453, 0.0, 0.0, 0.375, 0.0, 0.0, 0.0, 0.422, 0.0, 0.0]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestNRCExpandedEmotionFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = NRCExpandedEmotionFeaturizer()
        got = featurizer.featurize(['!'])
        expected = [0.0329545883908009, 0.10252551320880843, 0.0396174509579299, 0.02163596069596282,
                    0.18454179292881, 0.07689066037386351, 0.3002052242222958, 0.005777398957777484,
                    0.042446558283882, 0.03611382219459458]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestNRCHashtagEmotionFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = NRCHashtagEmotionFeaturizer()
        got = featurizer.featurize(['#badass'])
        expected = [0.00852973401896, 0.0, 0.0, 0.376417244806, 0.0, 0.0, 0.0, 0.0, 0.826006600008, 0.0]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestSentiStrengthFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = SentiStrengthFeaturizer()
        got = featurizer.featurize(['good', 'day'])
        expected = [2, -1]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestSentiWordNetFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = SentiWordNetFeaturizer()
        got = featurizer.featurize(['awesome'])
        expected = [0.875 - 0.125, 0.0]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestNegationFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = NegationFeaturizer()
        got = featurizer.featurize(['i', 'don\'t', 'like', 'it'])
        expected = [1]

        self.assertListEqual(
            expected,
            got,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )


class TestEdinburghEmbeddingFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = EdinburghEmbeddingsFeaturizer()
        got = featurizer.featurize(['i', 'don\'t', 'like', 'it'])
        self.assertTrue(len(got) == 100)


class TestEmojiEmbeddingFeaturizer(TestCase):
    def test_featurizer(self):
        featurizer = EmojiEmbeddingsFeaturizer()
        got = featurizer.featurize(['😂'])
        self.assertTrue(len(got) == 300)
