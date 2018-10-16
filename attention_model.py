from keras.layers import Bidirectional, GRU, RepeatVector, TimeDistributed, Dense, Flatten
from keras.models import Sequential

from ConcurrenceLayer import Concurrence


def create_model(hidden_size, input_size, max_out_seq_len, output_size):
    model = Sequential()
    model.add(Bidirectional(GRU(hidden_size, return_sequences=True), merge_mode='concat',
                            input_shape=(None, input_size)))  # Encoder  -> Paper says abc -> cba -> 123
    model.add(Concurrence())
    # model.add(RepeatVector(max_out_seq_len + 1))  # I don't like the "done" extra character
    model.add(RepeatVector(max_out_seq_len))
    model.add(GRU(hidden_size * 2, return_sequences=True))  # Decoder  # Maybe put here a classifier
    model.add(TimeDistributed(Dense(output_dim=output_size, activation="softmax")))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

    return model
