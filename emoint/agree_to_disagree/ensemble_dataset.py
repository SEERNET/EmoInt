from __future__ import print_function

import codecs
import csv
import pickle
import re

import nltk
import numpy as np
from tweetokenize import tokenizer

from emoint.featurizers.agree_to_disagree_featurizer import AgreeToDisagree
from emoint.featurizers.edinburgh_embeddings_featurizer import EdinburghEmbeddingsFeaturizer

def get_raw_data(file_path):
    print('Getting raw data')
    response = []
    quote = []
    category = []
    depths = []

    print('>>>>>>>>>>>>>')
    with open(file_path, 'r') as csvfile:
        print('>>>>>>>>>>>>>')
        reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL, delimiter='\t')
        print('>>>>>>>>>>>>>')
        for row in reader:
            response.append(row[1])
            quote.append(row[2])
            depths.append(int(row[3]))

            if row[0].lower().find('disagree') != -1:
                category.append(0)
            elif row[0].lower().find('agree') != -1:
                category.append(1)
            else:
                category.append(2)
    print('Number of Q-R pairs: %d' % len(response))
    return response, quote, category, depths


feat1 = AgreeToDisagree()
feat2 = EdinburghEmbeddingsFeaturizer(embedding_path='/home/venkatesh/glove.840B.300d.txt', dim=300)
tok = tokenizer.Tokenizer(allcapskeep=False)


def process(text):
    toks = tok.tokenize(text.decode('utf-8'))
    return ' '.join([re.sub('\'m|\'s', '', x) for x in toks]).encode('utf-8')


def get_data(response, quote, category, depths):
    print("Allocating space")
    lt = len(response)
    depths = np.array(depths)
    depths = np.expand_dims(depths, -1)
    rt = np.zeros(shape=(lt, feat1.dim + feat2.dim))
    qt = np.zeros(shape=(lt, feat1.dim + feat2.dim))
    print("Done Allocating space")

    for i, t in enumerate(response):
        t = process(t)
        rt[i] = np.append(
            feat1.featurize(t.decode('utf-8'), tok),
            feat2.featurize(t.decode('utf-8'), tok),
        )
        if i % 1000 == 0:
            print(i)

    for i, t in enumerate(quote):
        t = process(t)
        qt[i] = np.append(
            feat1.featurize(t.decode('utf-8'), tok),
            feat2.featurize(t.decode('utf-8'), tok)
        )
        if i % 1000 == 0:
            print(i)

    print(depths.shape, rt.shape, qt.shape)
    X = np.hstack((rt, qt, depths))
    category = np.array(category, dtype=int)
    return X, category


if __name__ == '__main__':
    MAX_NB_WORDS = 200000
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # dataset = 
    train_data = get_raw_data('/home/venkatesh/create_debate/train.txt'.format(id))
    dev_data = get_raw_data('/home/venkatesh/create_debate/dev.txt'.format(id))
    test_data = get_raw_data('/home/venkatesh/create_debate/test.txt'.format(id))

    train_data = get_data(train_data[0], train_data[1], train_data[2], train_data[3])
    dev_data = get_data(dev_data[0], dev_data[1], dev_data[2], dev_data[3])
    test_data = get_data(test_data[0], test_data[1], test_data[2], test_data[3])

    pickle.dump(train_data, open('/home/venkatesh/pkl_dir/ensemble_train_data.pkl', 'wb'))
    pickle.dump(dev_data, open('/home/venkatesh/pkl_dir/ensemble_dev_data.pkl', 'wb'))
    pickle.dump(test_data, open('/home/venkatesh/pkl_dir/ensemble_test_data.pkl', 'wb'))

