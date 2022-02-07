from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# import translation.project_tests as tests

# tokenize data
# turn each sentence into a sequence of words ids
def tokenize(x):
    x_tk = Tokenizer(char_level = False)

    # word -> index dictionary
    # create a dictionary of words and their id
    x_tk.fit_on_texts(x)

    # transform the text to a sequence of ids
    return x_tk.texts_to_sequences(x), x_tk


# all sequences must have the same length by adding a padding
def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])

    return pad_sequences(x, maxlen = length, padding = 'post')

# tests.test_pad(pad)

# preprocess data pipeline
# first tokenize and then add padding
def preprocess(sentences):
    lang_x, lang_y = sentences.keys()
    sent_x, sent_y = sentences.values()

    preprocess_x, x_tk = tokenize(sent_x)
    preprocess_y, y_tk = tokenize(sent_y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    max_x_sequence_length = preprocess_x.shape[1]
    max_y_sequence_length = preprocess_y.shape[1]

    x_vocab_size = len(x_tk.word_index)
    y_vocab_size = len(y_tk.word_index)

    print('Max {} sentence length:'.format(lang_x), max_x_sequence_length)
    print('Max {} sentence length:'.format(lang_y), max_y_sequence_length)
    print('{} vocabulary size:'.format(lang_x), x_vocab_size)
    print('{} vocabulary size:'.format(lang_y), y_vocab_size)

    return { lang_x: preprocess_x, lang_y: preprocess_y },\
        { lang_x: x_tk, lang_y: y_tk }