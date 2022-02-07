from os.path import join as path_join
from numpy import argmax as np_argmax

import translation.data_load as data_load
import translation.pre_process as pre_process
import translation.models as models

from keras.layers.embeddings import Embedding

sentences = { }

# initialize language object
def initialize_sentences(main, to):
    sentences[main] = []
    sentences[to] = []


# map word indexes to words
def logits_to_text(logits, tokenizer):
    index_to_words = { id: word for word, id in tokenizer.word_index.items() }
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np_argmax(logits, 1)])


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

        preproc_sentences, tokenizers, max_sequence_lengths, vocabulary_sizes = pre_process.preprocess(sentences)
        print('--- Data preprocessed')

        # train model
        simple_rnn_model, padded_data = models.simple_RNN_model(preproc_sentences, max_sequence_lengths, vocabulary_sizes)

        # print prediction(s)
        print(logits_to_text(simple_rnn_model.predict(padded_data[:1])[0], tokenizers[lang]))