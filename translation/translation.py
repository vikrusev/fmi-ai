from os.path import join as path_join

import translation.data_load as data_load
import translation.pre_process as pre_process

import numpy as np
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


sentences = { }

def initialize_sentences(main, to):
    sentences[main] = []
    sentences[to] = []

def translate(translate_from, *translate_to):
    for lang in translate_to:
        initialize_sentences(translate_from, lang)

        data_load.load_data(path_join('translation', 'vocabularies'), sentences)
        print('--- Dataset loaded\n')

        # print sample sentences
        print("--- Sample sentences")
        data_load.sample_sentences(sentences)

        # evaluate complexity of data
        print("--- Complexity of data")
        data_load.complexity_of_data(sentences)

        preproc_sentences, tokenizers = pre_process.preprocess(sentences)
        print('--- Data preprocessed')