from emoint.featurizers.base_featurizers import Featurizer
from emoint.featurizers.utils import senti_strength_jar_path, senti_strength_dir_path
import subprocess
import shlex
import re

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
        self._citation = 'Thelwall, Mike, et al. "Sentiment strength detection in short informal text." Journal of' \
                         ' the American Society for Information Science and Technology 61.12 (2010): 2544-2558.'

    @property
    def features(self):
        return [self.id + '-' + x for x in ['positive', 'negative']]

    def featurize(self, tokens):
        """This function returns sum of intensities of positive and negative tokens
        :param tokens list of tokens
        """
        p = subprocess.Popen(shlex.split("java -jar {} stdin sentidata {}".format(self.jar_path, self.dir_path)),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_text, stderr_text = p.communicate('+'.join(tokens).encode('utf-8'))
        stdout_text = re.sub('\s+', ' ', stdout_text)
        splits = stdout_text.rstrip().split(' ')
        return [
            sum(float(splits[i]) for i in xrange(0, len(splits), 2)),
            sum(float(splits[i]) for i in xrange(1, len(splits), 2))
        ]
