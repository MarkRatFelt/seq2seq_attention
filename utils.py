import os
from theano.gradient import np

from scipy import *


def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences


def find_checkpoint_file(folder):
    checkpoint_file = [folder + f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    return checkpoint_file

