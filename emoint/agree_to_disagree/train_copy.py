from __future__ import print_function

import errno
import json
import logging
import os
import pickle
import sys
import time

import keras
import numpy as np
import sklearn
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, GRU, dot, Flatten, Reshape, add, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.utils import to_categorical
from keras_diagram import ascii
from sklearn.metrics import f1_score, precision_score, recall_score


def list_files(base_path, predicate):
    for folder, subs, files in os.walk(base_path):
        for filename in files:
            if predicate(os.path.join(folder, filename)):
                yield (os.path.join(folder, filename))


def get_params(model):
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    logging.info('Total params: {:,}'.format(trainable_count + non_trainable_count))
    logging.info('Trainable params: {:,}'.format(trainable_count))
    logging.info('Non-trainable params: {:,}'.format(non_trainable_count))


def get_attention_model(nbw, wem, MAX_SEQUENCE_LENGTH=64,
                        WORD_EMBEDDING_DIM=300, SENT_EMBEDDING_DIM=64, DROPOUT=0.5, is_lex=True, nc=3):
    response = Input(shape=(MAX_SEQUENCE_LENGTH,))
    quote = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(response)
    q1 = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")(q1)

    q2 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(quote)
    q2 = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")(q2)

    dot_op = dot([q1, q2], [1, 1])
    attention = Flatten()(dot_op)
    attention = Dense((MAX_SEQUENCE_LENGTH * SENT_EMBEDDING_DIM))(attention)
    attention = Reshape((MAX_SEQUENCE_LENGTH, SENT_EMBEDDING_DIM))(attention)

    merged = add([q1, attention])
    merged = Flatten()(merged)

    if is_lex:
        lex_response = Input(shape=(133,))
        lex_quote = Input(shape=(133,))

        q3 = Dense(60, activation='relu')(lex_response)
        q4 = Dense(60, activation='relu')(lex_quote)

        merged = concatenate([merged, q3, q4])

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    category = Dense(nc, activation='softmax')(merged)

    if is_lex:
        model = Model(inputs=[response, quote, lex_response, lex_quote], outputs=category)
    else:
        model = Model(inputs=[response, quote], outputs=category)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    summ = ascii(model)
    print(summ)
    get_params(model)

    return model


def get_normal_model(nbw, wem, MAX_SEQUENCE_LENGTH=25,
                     WORD_EMBEDDING_DIM=300, DROPOUT=0.5, is_lex=True, nc=3):
    response = Input(shape=(MAX_SEQUENCE_LENGTH,))
    quote = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(response)
    q1 = TimeDistributed(Dense(WORD_EMBEDDING_DIM, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(WORD_EMBEDDING_DIM,))(q1)

    q2 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(quote)
    q2 = TimeDistributed(Dense(WORD_EMBEDDING_DIM, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(WORD_EMBEDDING_DIM,))(q2)

    if is_lex:
        lex_response = Input(shape=(133,))
        lex_quote = Input(shape=(133,))

        q3 = Dense(60, activation='relu')(lex_response)
        q4 = Dense(60, activation='relu')(lex_quote)

        merged = concatenate([q1, q2, q3, q4])
    else:
        merged = concatenate([q1, q2])

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    category = Dense(nc, activation='softmax')(merged)

    if is_lex:
        model = Model(inputs=[response, quote, lex_response, lex_quote], outputs=category)
    else:
        model = Model(inputs=[response, quote], outputs=category)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    summ = ascii(model)
    print(summ)
    get_params(model)

    return model


def get_rnn_model(nbw, wem, MAX_SEQUENCE_LENGTH=25,
                  WORD_EMBEDDING_DIM=300, SENT_EMBEDDING_DIM=64, DROPOUT=0.5, is_lex=True, nc=3):
    response = Input(shape=(MAX_SEQUENCE_LENGTH,))
    quote = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   # weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(response)
    q1 = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=False), merge_mode="sum")(q1)

    q2 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   # weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(quote)
    q2 = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=False), merge_mode="sum")(q2)

    if is_lex:
        lex_response = Input(shape=(133,))
        lex_quote = Input(shape=(133,))

        q3 = Dense(60, activation='relu')(lex_response)
        q4 = Dense(60, activation='relu')(lex_quote)

        merged = concatenate([q1, q2, q3, q4])
    else:
        merged = concatenate([q1, q2])

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    category = Dense(nc, activation='softmax')(merged)

    if is_lex:
        model = Model(inputs=[response, quote, lex_response, lex_quote], outputs=category)
    else:
        model = Model(inputs=[response, quote], outputs=category)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    summ = ascii(model)
    print(summ)
    get_params(model)

    return model


def get_shared(nbw, wem, WORD_EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, SENT_EMBEDDING_DIM, nc=3):
    model = keras.models.Sequential(
        layers=[
            Embedding(nbw + 1,
                      WORD_EMBEDDING_DIM,
                      weights=[wem],
                      input_length=MAX_SEQUENCE_LENGTH,
                      trainable=False, input_shape=(MAX_SEQUENCE_LENGTH,)),
            Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=False), merge_mode="sum")
        ]
    )
    return model


def get_siamese_rnn_model(nbw, wem, MAX_SEQUENCE_LENGTH=25,
                          WORD_EMBEDDING_DIM=300, SENT_EMBEDDING_DIM=64, DROPOUT=0.5, is_lex=True, nc=3):
    response = Input(shape=(MAX_SEQUENCE_LENGTH,))
    quote = Input(shape=(MAX_SEQUENCE_LENGTH,))

    shared_gru = get_shared(nbw, wem, WORD_EMBEDDING_DIM, MAX_SEQUENCE_LENGTH,
                            SENT_EMBEDDING_DIM)

    q1 = shared_gru(response)
    q2 = shared_gru(quote)

    if is_lex:
        lex_response = Input(shape=(133,))
        lex_quote = Input(shape=(133,))

        q3 = Dense(60, activation='relu')(lex_response)
        q4 = Dense(60, activation='relu')(lex_quote)

        merged = concatenate([q1, q2, q3, q4])
    else:
        merged = concatenate([q1, q2])

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    category = Dense(nc, activation='softmax')(merged)

    if is_lex:
        model = Model(inputs=[response, quote, lex_response, lex_quote], outputs=category)
    else:
        model = Model(inputs=[response, quote], outputs=category)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    summ = ascii(model)
    print(summ)
    get_params(model)

    return model


def get_rcnn(nbw, wem, MAX_SEQUENCE_LENGTH=25,
             WORD_EMBEDDING_DIM=300, SENT_EMBEDDING_DIM=64, DROPOUT=0.5, is_lex=True, nc=3):
    response = Input(shape=(MAX_SEQUENCE_LENGTH,))
    quote = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(response)
    q1 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', strides=1)(q1)
    q1 = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=False), merge_mode="sum")(q1)

    q2 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(quote)
    q2 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', strides=1)(q2)
    q2 = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=False), merge_mode="sum")(q2)

    if is_lex:
        lex_response = Input(shape=(133,))
        lex_quote = Input(shape=(133,))

        q3 = Dense(60, activation='relu')(lex_response)
        q4 = Dense(60, activation='relu')(lex_quote)

        merged = concatenate([q1, q2, q3, q4])
    else:
        merged = concatenate([q1, q2])

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    category = Dense(nc, activation='softmax')(merged)

    if is_lex:
        model = Model(inputs=[response, quote, lex_response, lex_quote], outputs=category)
    else:
        model = Model(inputs=[response, quote], outputs=category)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    summ = ascii(model)
    print(summ)
    get_params(model)

    return model


def get_cnn(nbw, wem, MAX_SEQUENCE_LENGTH=25,
            WORD_EMBEDDING_DIM=300, SENT_EMBEDDING_DIM=64, DROPOUT=0.5, is_lex=True, nc=3):
    response = Input(shape=(MAX_SEQUENCE_LENGTH,))
    quote = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(response)

    q1 = Conv1D(64, 5, activation='relu')(q1)
    q1 = Conv1D(64, 5, activation='relu')(q1)
    q1 = MaxPooling1D(3)(q1)
    q1 = Conv1D(128, 5, activation='relu')(q1)
    q1 = Conv1D(128, 5, activation='relu')(q1)
    q1 = GlobalAveragePooling1D()(q1)

    q2 = Embedding(nbw + 1,
                   WORD_EMBEDDING_DIM,
                   weights=[wem],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(quote)
    q2 = Conv1D(64, 5, activation='relu')(q2)
    q2 = Conv1D(64, 5, activation='relu')(q2)
    q2 = MaxPooling1D(3)(q2)
    q2 = Conv1D(128, 5, activation='relu')(q2)
    q2 = Conv1D(128, 5, activation='relu')(q2)
    q2 = MaxPooling1D(3)(q2)
    q2 = GlobalAveragePooling1D()(q2)

    if is_lex:
        lex_response = Input(shape=(133,))
        lex_quote = Input(shape=(133,))

        q3 = Dense(60, activation='relu')(lex_response)
        q4 = Dense(60, activation='relu')(lex_quote)

        merged = concatenate([q1, q2, q3, q4])
    else:
        merged = concatenate([q1, q2])

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    category = Dense(nc, activation='softmax')(merged)

    if is_lex:
        model = Model(inputs=[response, quote, lex_response, lex_quote], outputs=category)
    else:
        model = Model(inputs=[response, quote], outputs=category)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    summ = ascii(model)
    print(summ)
    get_params(model)

    return model


def mkdir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


class LoggingCallback(keras.callbacks.Callback):
    def __init__(self, print_fcn=logging.info):
        keras.callbacks.Callback.__init__(self)
        self.print_fcn = print_fcn
        self.cnt = 0

    def on_epoch_end(self, epoch, logs=None):
        self.cnt = 0
        self.print_fcn('Epoch: {}: {}'.format(epoch, logs))

    def on_batch_end(self, batch, logs=None):
        if self.cnt % 1000 == 0:
            self.print_fcn('Batch: {}: {}'.format(batch, logs))
        self.cnt += 1


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)

    def flush(self):
        pass


import uuid

if __name__ == '__main__':
    np.random.seed(0)

    dataset = 'awtp'
    # dataset = 'iacv2'

    """
    # 0 false local 32
    # 0 false local 64
    # 0 false local 128
    # 1 false local 32
    # 1 false local 64

    # 4 false local 64

    """
    model_num = int(sys.argv[1])
    lex = bool(int(sys.argv[2]))
    env = sys.argv[3]
    seq_len = int(sys.argv[4])

    if env == 'local':
        pred = '/home/venkatesh/pkl_dir/'
        log_dir = '/home/venkatesh/runs_dir/{}'.format(uuid.uuid4())
        fname = os.path.join(log_dir, "log.txt")
        mkdir(fname)
    else:
        pred = '/input/'
        log_dir = '/output/{}'.format(uuid.uuid4())
        fname = os.path.join(log_dir, "log.txt")
        mkdir(fname)

    format = '%(message)s'
    logging.basicConfig(level=logging.INFO, filename=fname, format=format)
    logger = logging.getLogger(__name__)
    inst = LoggerWriter(logger, logging.INFO)
    sys.stdout = inst
    # sys.stderr = inst

    logging.info(sys.argv)

    with open(os.path.join(pred, '{}_nb_words.json'.format(dataset)), 'r') as f_obj:
        nb_words = json.load(f_obj)['nb_words']
    word_embedding_matrix = np.load(open(os.path.join(pred, '{}_word_embedding_matrix.npy'.format(dataset)), 'rb'))

    train_r_32, train_q_32, train_r_64, train_q_64, train_r_128, train_q_128, \
    train_y, train_lex_r, train_lex_q = pickle.load(open(os.path.join(pred, '{}_train_data.pkl'.format(dataset)), 'rb'))

    dev_r_32, dev_q_32, dev_r_64, dev_q_64, dev_r_128, dev_q_128, \
    dev_y, dev_lex_r, dev_lex_q = pickle.load(open(os.path.join(pred, '{}_dev_data.pkl'.format(dataset)), 'rb'))

    test_r_32, test_q_32, test_r_64, test_q_64, test_r_128, test_q_128, \
    test_y, test_lex_r, test_lex_q = pickle.load(open(os.path.join(pred, '{}_test_data.pkl'.format(dataset)), 'rb'))

    t0 = time.time()

    batch_size = 32
    nc = 3

    if seq_len == 32:
        train_r, train_q, dev_r, dev_q, test_r, test_q = \
            train_r_32, train_q_32, dev_r_32, dev_q_32, test_r_32, test_q_32
    if seq_len == 64:
        train_r, train_q, dev_r, dev_q, test_r, test_q = \
            train_r_64, train_q_64, dev_r_64, dev_q_64, test_r_64, test_q_64
    if seq_len == 128:
        train_r, train_q, dev_r, dev_q, test_r, test_q = \
            train_r_128, train_q_128, dev_r_128, dev_q_128, test_r_128, test_q_128

    # train_rq = np.vstack([train_r, train_q])
    # train_qr = np.vstack([train_q, train_r])
    # train_yy = np.append(train_y, train_y)


    # ------------------

    train_yy = train_y
    train_rq = train_r
    train_qr = train_q

    dev_yy = dev_y
    dev_rq = dev_r
    dev_qr = dev_q

    test_yy = test_y
    test_rq = test_r
    test_qr = test_q

    # ------------------

    train_lex_rq = train_lex_r
    train_lex_qr = train_lex_q

    dev_lex_rq = dev_lex_r
    dev_lex_qr = dev_lex_q

    test_lex_rq = test_lex_r
    test_lex_qr = test_lex_q

    # ------------------

    selec = False
    if selec:
        nc = 2
        selection = np.where(train_yy != 2)
        train_yy = train_yy[selection]
        train_rq = train_rq[selection]
        train_qr = train_qr[selection]
        train_lex_rq = train_lex_rq[selection]
        train_lex_qr = train_lex_qr[selection]

        selection = np.where(dev_yy != 2)
        dev_yy = dev_yy[selection]
        dev_rq = dev_rq[selection]
        dev_qr = dev_qr[selection]
        dev_lex_rq = dev_lex_rq[selection]
        dev_lex_qr = dev_lex_qr[selection]

        selection = np.where(test_yy != 2)
        test_yy = test_yy[selection]
        test_rq = test_rq[selection]
        test_qr = test_qr[selection]
        test_lex_rq = test_lex_rq[selection]
        test_lex_qr = test_lex_qr[selection]

    print("Shapes: {}, {}".format(train_rq.shape, train_qr.shape))
    weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(train_yy), y=train_yy)
    logging.info('Class Weights: {}'.format(weights))

    if model_num == 0:
        mdl = get_normal_model(nb_words, word_embedding_matrix, DROPOUT=0.2,
                               MAX_SEQUENCE_LENGTH=seq_len, is_lex=lex, nc=nc)
    elif model_num == 1:
        mdl = get_rnn_model(nb_words, word_embedding_matrix, DROPOUT=0.2, SENT_EMBEDDING_DIM=128,
                            MAX_SEQUENCE_LENGTH=seq_len, is_lex=lex, nc=nc)
    elif model_num == 2:
        batch_size = 512
        mdl = get_attention_model(nb_words, word_embedding_matrix, DROPOUT=0.2,
                                  MAX_SEQUENCE_LENGTH=seq_len, is_lex=lex, nc=nc)

    elif model_num == 3:
        mdl = get_siamese_rnn_model(nb_words, word_embedding_matrix, DROPOUT=0.2,
                                    MAX_SEQUENCE_LENGTH=seq_len, is_lex=lex, nc=nc)
    elif model_num == 4:
        mdl = get_rcnn(nb_words, word_embedding_matrix, DROPOUT=0.2,
                       MAX_SEQUENCE_LENGTH=seq_len, is_lex=lex, nc=nc)
    else:
        mdl = get_cnn(nb_words, word_embedding_matrix, DROPOUT=0.2,
                      MAX_SEQUENCE_LENGTH=seq_len, is_lex=lex, nc=nc)

    cb = [
        ModelCheckpoint(
            os.path.join(log_dir, 'weights_{epoch:02d}_{val_loss:.2f}_{val_acc:.2f}.h5'),
            save_best_only=False
        ),
        LoggingCallback()
    ]

    if lex:

        mdl.fit([train_rq, train_qr, train_lex_rq, train_lex_qr],
                to_categorical(train_yy, num_classes=nc), epochs=20, class_weight=weights,
                validation_data=(
                    [dev_rq, dev_qr, dev_lex_rq, dev_lex_qr],
                    to_categorical(dev_yy, num_classes=nc)),
                verbose=0, batch_size=batch_size, callbacks=cb)

        files = list_files(log_dir, lambda x: x.endswith('.h5'))
        f_path = sorted(files, key=lambda x: float(os.path.split(x)[1].split('_')[2]))[0]
        print('Best validation file path: {}'.format(f_path))
        mdl.load_weights(filepath=f_path)

        met = mdl.evaluate(x=[test_rq, test_qr, test_lex_rq, test_lex_qr], verbose=0,
                           y=to_categorical(test_yy, num_classes=nc))

        test_pred = mdl.predict(x=[test_rq, test_qr, test_lex_rq, test_lex_qr])

    else:
        mdl.fit([train_rq, train_qr],
                to_categorical(train_yy, num_classes=nc), epochs=10, class_weight=weights,
                validation_data=(
                    [dev_rq, dev_qr],
                    to_categorical(dev_yy, num_classes=nc)),
                verbose=0, batch_size=batch_size, callbacks=cb)

        files = list_files(log_dir, lambda x: x.endswith('.h5'))
        f_path = sorted(files, key=lambda x: float(os.path.split(x)[1].split('_')[2]))[0]
        print('Best validation file path: {}'.format(f_path))
        mdl.load_weights(filepath=f_path)

        test_pred = mdl.predict(x=[test_rq, test_qr])

        met = mdl.evaluate(x=[test_rq, test_qr], verbose=0,
                           y=to_categorical(test_yy, num_classes=nc))

    logging.info("Keras metrics: {}".format(met))
    test_y_pred = np.argmax(test_pred, axis=1)
    if nc == 2:
        avg_l = ['binary']
    else:
        avg_l = [None, 'micro', 'macro', 'weighted']

    for avg in avg_l:
        fscore = f1_score(test_yy, test_y_pred, average=avg)
        precision = precision_score(test_yy, test_y_pred, average=avg)
        recall = recall_score(test_yy, test_y_pred, average=avg)

        print('---------{}---------'.format(avg))
        logging.info("Fscore: {}".format(fscore))
        logging.info("Precision: {}".format(precision))
        logging.info("Recall: {}".format(recall))
        print("------------------")


