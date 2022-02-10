from enum import Enum

from translation.models.rnn_simple import simple_rnn_model
from translation.models.rnn_embedding import embedding_rnn_model
from translation.models.rnn_bidirectional import bidirectional_rnn_model
from translation.models.rnn_encoder_decoder import encoder_decoder_rnn_model
from translation.models.rnn_embedding_bidirectional import embedding_bidirectional_rnn_model

class Models(Enum):
    SIMPLE = simple_rnn_model
    EMBEDDING = embedding_rnn_model
    BIDIRECTIONAL = bidirectional_rnn_model
    ENCODER_DECODER = encoder_decoder_rnn_model
    EMBEDDING_BIDIRECTIONAL = embedding_bidirectional_rnn_model