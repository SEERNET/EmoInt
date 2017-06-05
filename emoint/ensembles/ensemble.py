from collections import Counter
from glob import glob

import numpy as np
from scipy.stats.mstats import gmean


class Ensembler:
    def __init__(self):
        """
        This method instantiates `Ensembler` class.
        """
        return

    @staticmethod
    def deduce(l, weights, ensemble_type):
        """
        This function computes individual prediction based on type of ensemble
        :param ensemble_type: Type of ensemble technique. eg: "amean", "gmean", "vote" etc.
        :param weights: weights of predictions, used only in voting.
        :param l: list of individual predictions
        :return: prediction of ensemble
        """
        if ensemble_type == 'amean':
            return np.mean(np.array(l))
        elif ensemble_type == 'gmean':
            assert (np.array(l) > 0).all(), 'Geometric mean cannot be applied to negative numbers'
            return gmean(np.array(l))
        elif ensemble_type == 'vote':
            nl = []
            if weights is not None:
                for i, p in enumerate(l):
                    nl += [p] * weights[i]
            return Counter(l).most_common(1)[0][0]

    def ensemble(self, pred_files, ensemble_pred_file, weights=None, ensemble_type="amean"):
        """
        This method creates ensemble prediction.
        :param ensemble_type: Type of ensemble technique. eg: "amean", "gmean", "vote" etc.
        :param weights: weights of predictions, used only in voting. eg: [1, 2, 3]
        :param pred_files: regular expression of prediction files. eg: results/prediction*.csv
        :param ensemble_pred_file: file to write ensemble output to. eg:  results/ensemble_prediction.csv
        :return: None
        """

        supported = ["amean", "gmean", "vote"]
        assert ensemble_type in supported, "Unsupported ensemble type. Choose one from {}".format(supported)

        with open(ensemble_pred_file, "wb") as out_file:
            l = []
            files = sorted(glob(pred_files))
            if weights is not None:
                assert len(files) == len(weights), "Provide weights to all prediction files"
            for i, glob_file in enumerate(files):
                lines = [float(x) for x in open(glob_file).read().splitlines()]
                l.append(lines)
            zl = zip(*l)
            output = []
            for x in zl:
                output.append(self.deduce(x, weights, ensemble_type))
            for x in output:
                out_file.write("%s\n" % x)
