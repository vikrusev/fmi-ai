from os.path import join as path_join

import translation.data_load as data_load
import translation.project_tests as tests

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


sentences = { }

def initialize_sentences(main, secondaries):
    sentences[main] = []

    for lang in secondaries:
        sentences[lang] = []

def translate(translate_from, *translate_to):
    initialize_sentences(translate_from, translate_to)

    data_load.load_data(path_join('translation', 'vocabularies'), sentences)
    print('--- Dataset loaded\n')

    # print sample sentences
    print("--- Sample sentences\n")
    data_load.sample_sentences(sentences)
    print()

    # evaluate complexity of data
    print("--- Complexity of data\n")
    data_load.complexity_of_data(sentences)