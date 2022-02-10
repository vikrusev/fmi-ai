from translation.pre_process import pad

from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed
from keras.layers.embeddings import Embedding

from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# import translation.project_tests as tests

def embed_model(input_shape, vocab_size_y):
    learning_rate = 1e-3

    rnn = GRU(64, return_sequences=True, activation="tanh")
    embedding = Embedding(vocab_size_y, 64, input_length=input_shape[1]) 
    logits = TimeDistributed(Dense(vocab_size_y, activation="softmax"))

    model = Sequential()

    # embedding can only be used in first layer --> Keras Documentation
    model.add(embedding)
    model.add(rnn)
    model.add(logits)

    model.compile(loss = sparse_categorical_crossentropy,
                optimizer = Adam(learning_rate),
                metrics = ['accuracy'])

    return model

# tests.test_embed_model(embed_model)

def embedding_rnn_model(sentences, max_sequence_lengths, vocabulary_sizes):
    sent_x, sent_y = sentences.values()
    _, vocab_size_y = vocabulary_sizes.values()
    _, max_seq_len_y = max_sequence_lengths.values()

    tmp_x = pad(sent_x, max_seq_len_y)
    tmp_x = tmp_x.reshape((-1, sent_y.shape[-2]))

    model = embed_model(
        tmp_x.shape,
        vocab_size_y + 1
    )

    model.fit(tmp_x, sent_y, batch_size=1024, epochs=10, validation_split=0.2)

    return model, tmp_x