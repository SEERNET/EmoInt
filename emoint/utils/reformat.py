import gzip
import sys
from collections import defaultdict

"""
Utility functions to reformat downloaded resources
"""


def reformat(inp_path='emoint/resources/NRC-emotion-lexicon-wordlevel-v0.92.txt.gz',
             out_path='emoint/resources/NRC-emotion-lexicon-wordlevel-v0.92.txt.gz'):
    """
    Function to reformat data downloaded from http://saifmohammad.com/WebPages/lexicons.html
    :param inp_path: The input path where the data is present
    :param out_path: Optional.
    :return:
    """
    dd = defaultdict(lambda: defaultdict(float))
    data = gzip.open(inp_path).read().splitlines()

    for x in data:
        x = x.split('\t')
        dd[x[0]][x[1]] = float(x[2])

    with gzip.open(out_path, 'wb') as f:
        headers = dd.items()[0][1].keys()
        headers.sort()
        headers = ['word'] + headers
        f.write('\t'.join(headers) + '\n')

        for x in dd.keys():
            sorted_vals = dd[x].items()
            sorted_vals.sort()
            vals = [str(y[1]) for y in sorted_vals]
            vals = [x] + vals
            f.write('\t'.join(vals) + '\n')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        reformat(sys.argv[1])
    elif len(sys.argv) == 3:
        reformat(sys.argv[1], sys.argv[2])
    else:
        print("USAGE: python reformat.py /input/path/ /output/path/")
