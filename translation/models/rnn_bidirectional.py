from translation.pre_process import pad

from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Bidirectional

from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# import translation.project_tests as tests

def bidi_model(input_shape, vocab_size_y):
    learning_rate = 1e-3

    model = Sequential()

    model.add(Bidirectional(GRU(128, return_sequences = True, dropout = 0.1), 
                           input_shape = input_shape[1:]))
    model.add(TimeDistributed(Dense(vocab_size_y, activation = 'softmax')))

    model.compile(loss = sparse_categorical_crossentropy,
                 optimizer = Adam(learning_rate),
                 metrics = ['accuracy'])

    return model

# tests.test_bd_model(bidi_model)

def bidirectional_rnn_model(sentences, _, vocabulary_sizes):
    sent_x, sent_y = sentences.values()
    _, vocab_size_y = vocabulary_sizes.values()

    tmp_x = pad(sent_x, sent_y.shape[1])
    tmp_x = tmp_x.reshape((-1, sent_y.shape[-2], 1))

    model = bidi_model(
        tmp_x.shape,
        vocab_size_y + 1
    )

    model.fit(tmp_x, sent_y, batch_size=1024, epochs=20, validation_split=0.2)

    return model, tmp_x