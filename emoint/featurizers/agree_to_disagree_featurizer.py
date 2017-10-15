from emoint.featurizers.afinn_valence_featurizer import AFINNValenceFeaturizer
from emoint.featurizers.bing_liu_sentiment_featurizer import BingLiuFeaturizer
from emoint.featurizers.edinburgh_embeddings_featurizer import EdinburghEmbeddingsFeaturizer
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
from emoint.featurizers.base_featurizers import Featurizer
from emoint.featurizers.liwc_featurizer import LIWCFeaturizer


class AgreeToDisagree(Featurizer):
    @property
    def citation(self):
        # Todo
        return "PLACEHOLDER"

    @property
    def id(self):
        return "EmoInt"

    def __init__(self):
        self.featurizers = [
            AFINNValenceFeaturizer(),
            BingLiuFeaturizer(),
            MPQAEffectFeaturizer(),
            NRCAffectIntensityFeaturizer(),
            NRCEmotionFeaturizer(),
            NRCExpandedEmotionFeaturizer(),
            NRCHashtagEmotionFeaturizer(),
            NRCHashtagSentimentFeaturizer(),
            Sentiment140Featurizer(),
            SentiWordNetFeaturizer(),
            # SentiStrengthFeaturizer(),
            NegationFeaturizer(),
            # EdinburghEmbeddingsFeaturizer(),
            LIWCFeaturizer()
        ]
        self._features = self.collect_features(self.featurizers)

    @staticmethod
    def collect_features(featurizers):
        features = []
        for featurizer in featurizers:
            features += featurizer.features
        return features

    @property
    def features(self):
        return self._features

    @property
    def dim(self):
        """Dimension of feature space"""
        return len(self._features)

    def featurize(self, text, tokenizer):
        """Featurize using the following featurizers
            1. AFINNValenceFeaturizer
            2. BingLiuFeaturizer
            3. EdinburghEmbeddingsFeaturizer
            4. MPQAEffectFeaturizer
            5. NegationFeaturizer
            6. NRCAffectIntensityFeaturizer
            7. NRCEmotionFeaturizer
            8. NRCExpandedEmotionFeaturizer
            9. NRCHashtagEmotionFeaturizer
            10. NRCHashtagSentimentFeaturizer
            11. SentiWordNetFeaturizer
            12. Sentiment140Featurizer
            13. SentiStrengthFeaturizer
        """
        features = []
        for featurizer in self.featurizers:
            features += featurizer.featurize(text, tokenizer)
        return features



