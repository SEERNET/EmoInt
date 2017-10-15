from __future__ import print_function

import codecs
import csv
import json
import pickle
import re

import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tweetokenize import tokenizer

from emoint.featurizers.agree_to_disagree_featurizer import AgreeToDisagree

nltk.download('punkt')


def get_raw_data(file_path):
    print('Getting raw data')
    response = []
    quote = []
    category = []
    # depths = []

    with codecs.open(file_path) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL, delimiter='\t')
        for row in reader:
            response.append(row[1])
            quote.append(row[2])
            # depths.append(float(row[3]))

            if row[0].lower().find('disagree') != -1:
                category.append(0)
            elif row[0].lower().find('agree') != -1:
                category.append(1)
            else:
                category.append(2)
    print('Number of Q-R pairs: %d' % len(response))
    return response, quote, category#, depths


feat = AgreeToDisagree()
tok = tokenizer.Tokenizer(allcapskeep=False)


def process(text):
    toks = tok.tokenize(text.decode('utf-8'))
    return ' '.join([re.sub('\'m|\'s', '', x) for x in toks]).encode('utf-8')


def get_word_index(train_response, train_quote):
    print('Building Word Index')
    qr_paris = [process(x) for x in train_response] + [process(x) for x in train_quote]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(qr_paris)
    print("Words in index: %d" % len(tokenizer.word_index))
    return tokenizer, tokenizer.word_index


def get_embeddings_index(file_path):
    embeddings_index = {}
    with open(file_path) as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    print('Number of Word embeddings: %d' % len(embeddings_index))
    return embeddings_index


def get_word_embedding_matrix(word_index, embeddings_index, MAX_NB_WORDS, EMBEDDING_DIM=300):
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

    print(">>>>>>>>>>>>>>>>>>>>>")
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
        else:
            print(word)
    print(">>>>>>>>>>>>>>>>>>>>>")
    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
    print('Not Null embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) != 0))
    return nb_words, word_embedding_matrix


# def get_data(response, quote, category, depths, tokenizer, MAX_SEQUENCE_LENGTH=25):
def get_data(response, quote, category, tokenizer, MAX_SEQUENCE_LENGTH=25):
    response_seq = tokenizer.texts_to_sequences(response)
    quote_seq = tokenizer.texts_to_sequences(quote)

    padded_response_seq_32 = pad_sequences(response_seq, maxlen=32)
    padded_quote_seq_32 = pad_sequences(quote_seq, maxlen=32)

    padded_response_seq_64 = pad_sequences(response_seq, maxlen=64)
    padded_quote_seq_64 = pad_sequences(quote_seq, maxlen=64)

    padded_response_seq_128 = pad_sequences(response_seq, maxlen=128)
    padded_quote_seq_128 = pad_sequences(quote_seq, maxlen=128)

    category = np.array(category, dtype=int)

    print("Allocating space")
    lt = len(response)
    rt = np.zeros(shape=(lt, feat.dim))
    qt = np.zeros(shape=(lt, feat.dim))
    print("Done Allocating space")

    for i, t in enumerate(response):
        rt[i] = feat.featurize(t.decode('utf-8'), tok)
        if i % 1000 == 0:
            print(i)

    for i, t in enumerate(quote):
        qt[i] = feat.featurize(t.decode('utf-8'), tok)
        if i % 1000 == 0:
            print(i)

    return padded_response_seq_32, padded_quote_seq_32, \
           padded_response_seq_64, padded_quote_seq_64, \
           padded_response_seq_128, padded_quote_seq_128, \
           category, rt, qt#, np.array(depths, dtype='float32')

# 
# if __name__ == '__main__':
#     MAX_NB_WORDS = 200000
# 
#     train_data = get_raw_data('/home/venkatesh/create_debate/train.txt'.format(id))
#     dev_data = get_raw_data('/home/venkatesh/create_debate/dev.txt'.format(id))
#     test_data = get_raw_data('/home/venkatesh/create_debate/test.txt'.format(id))
# 
#     print(len(train_data), len(dev_data), len(test_data))
# 
#     tokenizer, word_index = get_word_index(train_data[0], train_data[1])
#     pickle.dump(tokenizer, open('/home/venkatesh/pkl_dir/tokenizer.pkl', 'wb'))
#     pickle.dump(word_index, open('/home/venkatesh/pkl_dir/word_index.pkl', 'wb'))
# 
#     embeddings_index = get_embeddings_index('/home/venkatesh/glove.840B.300d.txt')
#     nb_words, word_embedding_matrix = get_word_embedding_matrix(word_index, embeddings_index, MAX_NB_WORDS)
# 
#     train_data = get_data(train_data[0], train_data[1], train_data[2], train_data[3], tokenizer)
#     dev_data = get_data(dev_data[0], dev_data[1], dev_data[2], dev_data[3], tokenizer)
#     test_data = get_data(test_data[0], test_data[1], test_data[2], test_data[3], tokenizer)
# 
#     pickle.dump(train_data, open('/home/venkatesh/pkl_dir/train_data.pkl', 'wb'))
#     pickle.dump(dev_data, open('/home/venkatesh/pkl_dir/dev_data.pkl', 'wb'))
#     pickle.dump(test_data, open('/home/venkatesh/pkl_dir/test_data.pkl', 'wb'))
#     np.save(open('/home/venkatesh/pkl_dir/word_embedding_matrix.npy', 'wb'), word_embedding_matrix)
#     with open('/home/venkatesh/pkl_dir/nb_words.json', 'w') as f_obj:
#         json.dump({'nb_words': nb_words}, f_obj)
# 
# 

## iacv2 data
# if __name__ == '__main__':
#     MAX_NB_WORDS = 200000
# 
#     train_data = get_raw_data('/home/venkatesh/iacv2_data.txt'.format(id))
#     tokenizer = pickle.load(open('/home/venkatesh/pkl_dir/tokenizer.pkl', 'rb'))
# 
#     train_data = get_data(train_data[0], train_data[1], train_data[2], train_data[3], tokenizer)
#     pickle.dump(train_data, open('/home/venkatesh/pkl_dir/iacv2_train_data.pkl', 'wb'))
# 

## awtp data 1
# if __name__ == '__main__':
#     MAX_NB_WORDS = 200000
# 
#     train_data = get_raw_data('/home/venkatesh/agreement_annotations/packaged/merged.txt')
#     tokenizer = pickle.load(open('/home/venkatesh/pkl_dir/tokenizer.pkl', 'rb'))
# 
#     train_data = get_data(train_data[0], train_data[1], train_data[2], tokenizer)
#     pickle.dump(train_data, open('/home/venkatesh/pkl_dir/awtp_merged.pkl', 'wb'))
# 
# 


## awtp data 2
if __name__ == '__main__':
    MAX_NB_WORDS = 200000
    dataset = 'awtp'

    # train_data = get_raw_data('/home/venkatesh/iacv1_train.txt')
    # dev_data = get_raw_data('/home/venkatesh/iacv1_dev.txt')
    # test_data = get_raw_data('/home/venkatesh/iacv1_test.txt')

    train_data = get_raw_data('/home/venkatesh/agreement_annotations/packaged/train.txt')
    dev_data = get_raw_data('/home/venkatesh/agreement_annotations/packaged/dev.txt')
    test_data = get_raw_data('/home/venkatesh/agreement_annotations/packaged/test.txt')


    print(len(train_data), len(dev_data), len(test_data))

    tokenizer, word_index = get_word_index(train_data[0], train_data[1])
    pickle.dump(tokenizer, open('/home/venkatesh/pkl_dir/{}_tokenizer.pkl'.format(dataset), 'wb'))
    pickle.dump(word_index, open('/home/venkatesh/pkl_dir/{}_word_index.pkl'.format(dataset), 'wb'))

    embeddings_index = get_embeddings_index('/home/venkatesh/glove.840B.300d.txt')
    nb_words, word_embedding_matrix = get_word_embedding_matrix(word_index, embeddings_index, MAX_NB_WORDS)

    train_data = get_data(train_data[0], train_data[1], train_data[2], tokenizer)
    dev_data = get_data(dev_data[0], dev_data[1], dev_data[2], tokenizer)
    test_data = get_data(test_data[0], test_data[1], test_data[2], tokenizer)

    pickle.dump(train_data, open('/home/venkatesh/pkl_dir/{}_train_data.pkl'.format(dataset), 'wb'))
    pickle.dump(dev_data, open('/home/venkatesh/pkl_dir/{}_dev_data.pkl'.format(dataset), 'wb'))
    pickle.dump(test_data, open('/home/venkatesh/pkl_dir/{}_test_data.pkl'.format(dataset), 'wb'))
    np.save(open('/home/venkatesh/pkl_dir/{}_word_embedding_matrix.npy'.format(dataset), 'wb'), word_embedding_matrix)
    with open('/home/venkatesh/pkl_dir/{}_nb_words.json'.format(dataset), 'w') as f_obj:
        json.dump({'nb_words': nb_words}, f_obj)



