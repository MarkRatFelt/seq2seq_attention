import argparse

import keras
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from nltk import FreqDist, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from attention_model import create_model
from utils import process_data, find_checkpoint_file

ap = argparse.ArgumentParser()
ap.add_argument('-max_len', type=int, default=100)
ap.add_argument('-vocab_size', type=int, default=100000)
ap.add_argument('-batch_size', type=int, default=128)
ap.add_argument('-layer_num', type=int, default=1)
ap.add_argument('-hidden_dim', type=int, default=128)
ap.add_argument('-nb_epoch', type=int, default=100)
ap.add_argument('-mode', default='train-test')
args = vars(ap.parse_args())

MAX_LEN = args['max_len']
VOCAB_SIZE = args['vocab_size']
BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
HIDDEN_DIM = args['hidden_dim']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']

seed = 155
np.random.seed(seed)

data_set = pd.read_csv(
    "data/train_arithmetic.tsv",
    header=None).values

X_train, X_test, Y_train, Y_test = train_test_split(data_set[:, 0:],
                                                    [str(i[0].split("\t", 1)[1]) for i in data_set[:]],
                                                    test_size=0.3, random_state=87)

X_train = np.array([list(x[0]) for x in X_train])
X_test = np.array([list(x[0]) for x in X_test])
Y_train = np.array([list(x) for x in Y_train])
Y_test = np.array([list(x) for x in Y_test])

X_vocab = FreqDist(np.hstack(np.concatenate((X_train, X_test), axis=0)))
Y_vocab = FreqDist(np.hstack(np.concatenate((Y_train, Y_test), axis=0)))

X_ix_to_char = [char[0] for char in X_vocab]
Y_ix_to_char = [char[0] for char in Y_vocab]
X_ix_to_char.insert(0, 'ZERO')
Y_ix_to_char.insert(0, 'ZERO')
X_ix_to_char.append('UNK')
Y_ix_to_char.append('UNK')

X_test_ix_to_char = [char[0] for char in X_vocab]
Y_test_ix_to_char = [char[0] for char in Y_vocab]
X_test_ix_to_char.insert(0, 'ZERO')
Y_test_ix_to_char.insert(0, 'ZERO')
X_test_ix_to_char.append('UNK')
Y_test_ix_to_char.append('UNK')

X_char_to_ix = {word: ix for ix, word in enumerate(X_ix_to_char)}
Y_char_to_ix = {word: ix for ix, word in enumerate(Y_ix_to_char)}

X_test_char_ix = {word: ix for ix, word in enumerate(X_test_ix_to_char)}
Y_test_char_ix = {word: ix for ix, word in enumerate(Y_test_ix_to_char)}

for i, sentence in enumerate(X_train):
    for j, char in enumerate(sentence):
        if char in X_char_to_ix:
            X_train[i][j] = X_char_to_ix[char]
        else:
            X_train[i][j] = X_char_to_ix['UNK']

for i, sentence in enumerate(Y_train):
    for j, char in enumerate(sentence):
        if char in Y_char_to_ix:
            Y_train[i][j] = Y_char_to_ix[char]
        else:
            Y_train[i][j] = Y_char_to_ix['UNK']

for i, sentence in enumerate(X_test):
    for j, char in enumerate(sentence):
        if char in X_test_char_ix:
            X_test[i][j] = X_test_char_ix[char]
        else:
            X_test[i][j] = X_test_char_ix['UNK']

for i, sentence in enumerate(Y_test):
    for j, char in enumerate(sentence):
        if char in Y_test_char_ix:
            Y_test[i][j] = Y_test_char_ix[char]
        else:
            Y_test[i][j] = Y_test_char_ix['UNK']

X_max_len = max([len(sentence) for sentence in X_train])
y_max_len = max([len(sentence) for sentence in Y_train])

X_test_max_len = max([len(sentence) for sentence in X_test])
y_test_max_len = max([len(sentence) for sentence in Y_test])

X_train = pad_sequences(X_train, maxlen=X_max_len, dtype='int32')
Y_train = pad_sequences(Y_train, maxlen=y_max_len, dtype='int32')

X_test = pad_sequences(X_test, maxlen=X_test_max_len, dtype='int32')
Y_test = pad_sequences(Y_test, maxlen=y_test_max_len, dtype='int32')

model = create_model(HIDDEN_DIM, len(X_vocab) + 2, y_max_len, len(Y_vocab) + 2)

early_stop_criteria = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=2, verbose=0, mode='auto')
scaler = StandardScaler()

saved_weights = find_checkpoint_file('model_hdf5/')

if MODE == 'train':
    k_start = 1

    # If any trained weight was found, then load them into the model
    if len(saved_weights) != 0:
        print('[INFO] Saved weights found, loading...')
        epoch = saved_weights[saved_weights.rfind('_') + 1:saved_weights.rfind('.')]
        model.load_weights(saved_weights)
        k_start = int(epoch) + 1

    i_end = 0
    for k in range(k_start, NB_EPOCH + 1):
        print(k)
        # Shuffling the training data every epoch to avoid local minima
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        # Training 1000 sequences at a time
        for i in range(0, len(X_train), 1000):
            if i + 1000 >= len(X_train):
                i_end = len(X_train)
            else:
                i_end = i + 1000
            y_sequences = process_data(Y_train[i:i_end], y_max_len, Y_char_to_ix)
            x_sequences = process_data(X_train[i:i_end], X_max_len, X_char_to_ix)

            print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X_train)))
            model.fit(x_sequences, y_sequences, batch_size=BATCH_SIZE, epochs=1,
                      validation_split=0.2, verbose=1)
        model.save_weights('model_hdf5/checkpoint_epoch_{}.hdf5'.format(k))

elif MODE == 'train-test':
    x_sequences = process_data(X_train, X_max_len, X_char_to_ix)
    y_sequences = process_data(Y_train, y_max_len, Y_char_to_ix)

    x_test_sequences = process_data(X_test, X_test_max_len, X_test_char_ix)  # Maybe I should fix this
    y_test_sequences = process_data(Y_test, y_test_max_len, Y_test_char_ix)

    model_output = model.fit(x_sequences, y_sequences, epochs=1000, verbose=1, batch_size=BATCH_SIZE,
                             initial_epoch=0, callbacks=[early_stop_criteria], validation_split=0.2)

    model.save_weights('model_hdf5/checkpoint_epoch_{}.hdf5'.format(1000))

    output = {}

    validation_scores = model.evaluate(model_output.validation_data[0],
                                       model_output.validation_data[1], verbose=0)
    validation_size = model_output.validation_data[0].shape[0]
    output['validation_loss'] = validation_scores[0]
    output['validation_acc'] = validation_scores[1]
    training_size = X_train.shape[0] - validation_size

    test_scores = model.evaluate(x_test_sequences, y_test_sequences, verbose=0)

    output['test_loss'] = test_scores[0]
    output['test_acc'] = test_scores[1]

    print(output)
