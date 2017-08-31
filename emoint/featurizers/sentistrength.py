from emoint.featurizers.base_featurizers import Featurizer
from emoint.featurizers.utils import senti_strength_jar_path, senti_strength_dir_path
import os

"""
Info: http://sentistrength.wlv.ac.uk/
"""


class SentiStrengthFeaturizer(Featurizer):
    """SentiStrength Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def citation(self):
        return self._citation

    def __init__(self, jar_path=senti_strength_jar_path, dir_path=senti_strength_dir_path):
        """Initialize SentiStrength featurizer
        :param jar_path path to SentiStrength jar
        :param dir_path path to SentiStrength data directory
        """
        self._id = 'SentiStrength'
        self.jar_path = jar_path
        self.dir_path = dir_path

        if 'CLASSPATH' in os.environ:
            os.environ['CLASSPATH'] += ":" + jar_path
        else:
            os.environ['CLASSPATH'] = jar_path

        # Add jar to class path
        # Create and initialize the SentiStrength class
        from jnius import autoclass

        self.senti_obj = autoclass('uk.ac.wlv.sentistrength.SentiStrength')()
        self.senti_obj.initialise(["sentidata", senti_strength_dir_path,"trinary"])

        self._citation = 'Thelwall, Mike, et al. "Sentiment strength detection in short informal text." Journal of' \
                         ' the American Society for Information Science and Technology 61.12 (2010): 2544-2558.'

    @property
    def features(self):
        return [self.id + '-' + x for x in ['positive', 'negative']]

    def featurize(self, text, tokenizer):
        """This function returns sum of intensities of positive and negative tokens
        :param text text to featurize
        :param tokenizer tokenizer to tokenize text
        """
        tokens = tokenizer.tokenize(text)
        data = '+'.join(tokens).encode('utf-8').decode("utf-8", "ignore")
        score = self.senti_obj.computeSentimentScores(data)
        splits = score.rstrip().split(' ')
        return [float(splits[0]), float(splits[1])]
