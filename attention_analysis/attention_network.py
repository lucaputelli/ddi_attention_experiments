from keras.layers import Input, Concatenate, Bidirectional, LSTM, Dense
from keras.models import Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from attention_extraction_layers import AttentionWeights, ContextVector


def channel_inputs(dim, name: str, pos_tag: bool, offset: bool):
    input_list = list()
    word = Input(shape=(dim, 200), name=name+'_word')
    input_list.append(word)
    if pos_tag:
        pos = Input(shape=(dim, 4), name=name+'_pos')
        input_list.append(pos)
    if offset:
        d1 = Input(shape=(dim, 1), name=name+'_d1')
        d2 = Input(shape=(dim, 1), name=name+'_d2')
        input_list.append(d1)
        input_list.append(d2)
    if len(input_list) > 1:
        channel_input = Concatenate()(input_list)
    else:
        channel_input = input_list[0]
    return input_list, channel_input


def attention_model(max_length: int, pos_tag: bool, offset: bool, dimension: int, dropout: float,
                                   recurrent_dropout: float, learning_rate: float) -> Model:

    input_list, channel_input = channel_inputs(max_length, 'sent', pos_tag, offset)
    lstm = Bidirectional(LSTM(dimension, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                              dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True),
                         name='document_lstm')(channel_input)
    attention_weights = AttentionWeights(max_length, name='attention_weights')(lstm)
    context_vector = ContextVector()([lstm, attention_weights])
    output = Dense(5, activation='softmax', name='main_output')(context_vector)
    model = Model(inputs=input_list, outputs=[output])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate, decay=0, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    model.summary()
    return model


def get_weights_model(trained_model: Model) -> Model:
    intermediate_layer_model = Model(inputs=trained_model.input,
                                     outputs=[trained_model.get_layer('attention_weights').output])
    return intermediate_layer_model
