import shutil
import tempfile
import unittest
from os import path

from emoint.ensembles.ensemble import Ensembler


class TestEnsembles(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ensembler = Ensembler()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def helper(self, func, ens_type, out_file, expected, w=None):
        for i in range(3):
            with open(path.join(self.test_dir, 'prediction_{}.txt'.format(i)), 'w') as f_obj:
                for j in range(5):
                    f_obj.write("{}\n".format(func(i + j)))

        out_file = path.join(self.test_dir, out_file)
        self.ensembler.ensemble('{}/prediction*'.format(self.test_dir), out_file, ensemble_type=ens_type, weights=w)
        with open(out_file, 'r') as f:
            got = f.read().splitlines()
        self.assertListEqual(
            [round(float(x), 1) for x in got],
            expected,
            msg='Expected: {} != Got: {}'.format(expected, got)
        )

    def test_amean_ensemble(self):
        self.helper(lambda x: x, "amean", "ensemble_amean.txt", [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_gmean_ensemble(self):
        self.helper(lambda x: 2 ** x, "gmean", "ensemble_amean.txt", [2.0, 4.0, 8.0, 16.0, 32.0])

    def test_vote_ensemble(self):
        self.helper(lambda x: x, "vote", "ensemble_vote.txt", [0.0, 1.0, 2.0, 3.0, 4.0], [3, 1, 1])
