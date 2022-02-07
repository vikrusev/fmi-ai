from translation.pre_process import pad

from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation
from keras.losses import sparse_categorical_crossentropy

from tensorflow.keras.optimizers import Adam

# import translation.project_tests as tests

def simple_model(input_shape, vocab_size_y):
    learning_rate = 1e-3

    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences = True)(input_seq)
    logits = TimeDistributed(Dense(vocab_size_y))(rnn)

    model = Model(input_seq, Activation('softmax')(logits))

    model.compile(loss = sparse_categorical_crossentropy,
                optimizer = Adam(learning_rate),
                metrics = ['accuracy'])

    return model

# tests.test_simple_model(simple_model)

def simple_rnn_model(sentences, max_sequence_lengths, vocabulary_sizes):
    sent_x, sent_y = sentences.values()
    _, vocab_size_y = vocabulary_sizes.values()
    _, max_seq_len_y = max_sequence_lengths.values()

    tmp_x = pad(sent_x, max_seq_len_y)
    tmp_x = tmp_x.reshape((-1, sent_y.shape[-2], 1))

    # train the neural network
    model = simple_model(
        tmp_x.shape,
        vocab_size_y + 1
    )

    model.fit(tmp_x, sent_y, batch_size=1024, epochs=10, validation_split=0.2)

    return model, tmp_x