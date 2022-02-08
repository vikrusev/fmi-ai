from translation.pre_process import pad

from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding

from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# import translation.project_tests as tests

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    model = Sequential()

    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))

    learning_rate = 0.005

    model.compile(loss = sparse_categorical_crossentropy,
                 optimizer = Adam(learning_rate),
                 metrics = ['accuracy'])

    return model

# tests.test_model_final(model_final)

def embedding_bidirectional_rnn_model(sentences, _, vocabulary_sizes):
    sent_x, sent_y = sentences.values()
    vocab_size_x, vocab_size_y = vocabulary_sizes.values()

    tmp_x = pad(sent_x)
    model = model_final(tmp_x.shape,
        sent_y.shape[1],
        vocab_size_x + 1,
        vocab_size_y + 1
    )

    model.fit(tmp_x, sent_y, batch_size = 1024, epochs = 17, validation_split = 0.2)

    return model, tmp_x