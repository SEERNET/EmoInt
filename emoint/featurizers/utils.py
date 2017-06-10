from os.path import join
from os.path import dirname


def get_bigrams(tokens):
    return [a + " " + b for a, b in zip(tokens, tokens[1:])]


def resource_path():
    return join(dirname(__file__), '..', 'resources')

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

# resource paths
afinn_lexicon_path = join(resource_path(), 'AFINN-en-165.txt.gz')
afinn_emoticon_path = join(resource_path(), 'AFINN-emoticon-8.txt.gz')
bing_liu_lexicon_path = join(resource_path(), 'BingLiu.txt.gz')
mpqa_lexicon_path = join(resource_path(), 'mpqa.txt.gz')
nrc_affect_intensity_lexicon_path = join(resource_path(), 'nrc_affect_intensity.txt.gz')
nrc_emotion_lexicon_path = join(resource_path(), 'NRC-emotion-lexicon-wordlevel-v0.92.txt.gz')
nrc_hashtag_sentiment_unigram_lexicon_path = join(resource_path(), 'NRC-Hashtag-Sentiment-Lexicon-v0.1',
                                                  'unigrams-pmilexicon.txt.gz')
nrc_hashtag_sentiment_bigram_lexicon_path = join(resource_path(), 'NRC-Hashtag-Sentiment-Lexicon-v0.1',
                                                 'bigrams-pmilexicon.txt.gz')
sentiment140_unigram_lexicon_path = join(resource_path(), 'Sentiment140-Lexicon-v0.1', 'unigrams-pmilexicon.txt.gz')
sentiment140_bigram_lexicon_path = join(resource_path(), 'Sentiment140-Lexicon-v0.1', 'bigrams-pmilexicon.txt.gz')
nrc_expanded_emotion_lexicon_path = join(resource_path(), 'w2v-dp-BCC-Lex.txt.gz')
nrc_hashtag_emotion_lexicon_path = join(resource_path(), 'NRC-Hashtag-Emotion-Lexicon-v0.2.txt.gz')
senti_strength_jar_path = join(resource_path(), 'SentiStrength.jar')
senti_strength_dir_path = join(resource_path(), 'SentiStrength/')
senti_wordnet_lexicon_path = join(resource_path(), 'SentiWordNet_3.0.0.txt.gz')
negation_lexicon_path = join(resource_path(), 'NegatingWordList.txt.gz')
edinburgh_embedding_path = join(resource_path(), 'w2v.twitter.edinburgh.100d.csv.gz')
emoji_embedding_path = join(resource_path(), 'emoji2vec.txt.gz')
emoint_data = join(resource_path(), 'emoint/')
