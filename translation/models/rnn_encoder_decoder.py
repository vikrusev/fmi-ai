from translation.pre_process import pad

from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector

from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# import translation.project_tests as tests

def encdec_model(input_shape, output_sequence_length, vocab_size_y):
    learning_rate = 1e-3

    model = Sequential()

    model.add(GRU(128, input_shape = input_shape[1:], return_sequences = False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(128, return_sequences = True))
    model.add(TimeDistributed(Dense(vocab_size_y, activation = 'softmax')))

    model.compile(loss = sparse_categorical_crossentropy,
                optimizer = Adam(learning_rate),
                metrics = ['accuracy'])

    return model

# tests.test_encdec_model(encdec_model)

def encoder_decoder_rnn_model(sentences, _, vocabulary_sizes):
    sent_x, sent_y = sentences.values()
    _, vocab_size_y = vocabulary_sizes.values()

    tmp_x = pad(sent_x)
    tmp_x = tmp_x.reshape((-1, sent_x.shape[1], 1))

    model = encdec_model(
        tmp_x.shape,
        sent_y.shape[1],
        vocab_size_y + 1
    )

    model.fit(tmp_x, sent_y, batch_size=1024, epochs=20, validation_split=0.2)

    return model, tmp_x