from os.path import join as path_join
import numpy as np

import translation.data_load as data_load
import translation.pre_process as pre_process

from keras.preprocessing.sequence import pad_sequences

sentences = { }

# initialize language object
def initialize_sentences(main, to):
    sentences[main] = []
    sentences[to] = []


# map word indexes to words
def logits_to_text(logits, tokenizer):
    index_to_words = { id: word for word, id in tokenizer.word_index.items() }
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


# predict actual sentences
def final_predictions(model, sentences, tokenizers):
    x, y = sentences.values()
    x_tk, y_tk = tokenizers.values()

    y_id_to_word = { value: key for key, value in y_tk.word_index.items() }
    y_id_to_word[0] = '<PAD>'

    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])

    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')

    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


# main function
def translate(chosen_model, translate_from, *translate_to):
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
        rnn_model, padded_data = chosen_model(preproc_sentences, max_sequence_lengths, vocabulary_sizes)

        # print prediction(s)
        print(logits_to_text(rnn_model.predict(padded_data[:1])[0], tokenizers[lang]))

        final_predictions(rnn_model, preproc_sentences, tokenizers)