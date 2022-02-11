from os.path import join as path_join
import numpy as np

from models_enum import Models
import translation.file_helper as file_helper

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


# generate_translated
def predict_translations(srt_file_path, maxlen, model, x_tk):
    sentences_raw, timings = file_helper.get_data_from_file(srt_file_path)
    preproc_sentences = []

    # preprocess sentences to translate
    for sentence in sentences_raw:
        sentence = [x_tk.word_index[word] if word in x_tk.word_index else 1 for word in sentence.split()]
        sentence = pad_sequences([sentence], maxlen=maxlen, padding='post')
        preproc_sentences.append(sentence[0])

    # build array for translation
    sentences_translate = np.array([preproc_sentences[i] for i in range(0, len(preproc_sentences))])

    # predict all sentences
    return model.predict(sentences_translate, len(sentences_translate)), timings


# main function
def translate(srt_file_path, chosen_model, translate_from, *translate_to):
    for lang in translate_to:
        print('--- Start translating from {} to {}'.format(translate_from, lang))

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
        # print(logits_to_text(rnn_model.predict(padded_data[:1])[0], tokenizers[lang]))

        filepath_save = '{}_{}.srt'.format(srt_file_path, lang)
        x_tk, y_tk = tokenizers.values()
        x_sent, _ = preproc_sentences.values()
        if chosen_model == Models.EMBEDDING_BIDIRECTIONAL:
            predictions, timings = predict_translations('{}.srt'.format(srt_file_path), x_sent.shape[-1], rnn_model, x_tk)
            file_helper.build_translated_file(filepath_save, predictions, timings, y_tk)

        print('------ Translation from {} to {} completed!'.format(translate_from, lang))
        print('------ Subtitles saved to {}'.format(filepath_save))