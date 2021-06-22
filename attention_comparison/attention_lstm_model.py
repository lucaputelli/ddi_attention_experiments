from gensim.models import Word2Vec
from keras.layers import concatenate, Dense, LSTM, Bidirectional, Input, Embedding, \
    Flatten,average
from keras.models import Model
from keras.callbacks import ModelCheckpoint, History
from keras.initializers import *
from keras.optimizers import *
from keras_self_attention import SeqSelfAttention
from Attention import CandidateAttention
from AttentionWithContext import AttentionWithContext
from typing import Iterable, Tuple
from post_processing_lib import *
from AttentionMechanism import AttentionL
import os
from input_strings import *
from input_configurations import only_sentence, WORD, POS_TAG, OFFSET
from scipy import stats


class AttentionDDIModel:

    def __init__(self, folder: str, sentence_max_len: int, off_dim: int, word_model: Word2Vec,
                 tag_model: Word2Vec, hyperparameters_dist: dict):
        self.folder = folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.sentence_max_len = sentence_max_len
        self.off_dim = off_dim
        self.word_model = word_model
        self.tag_model = tag_model
        self.histories = list()
        self.hyperparameters_dist = hyperparameters_dist
        self.hyperparameters_values = {}
        self.checkpoint = None
        self.input_config = only_sentence
        self.attention = False
        self.word_attention = True

    def confronto_attention(self, training_set, training_labels, validation_set, validation_labels,
                             test_set, negative_classes, total_instances, total_classes, model_number: int):
        layers = 1
        epochs = 40
        attention_choices = ['att', 'self_att', 'context_att']
        for i in range(model_number):
            p = self.hyperparameters_dist
            dimension = p['dimension'].rvs() * 20
            dropout = p['dropout'].rvs() / 100
            recurrent_dropout = p['recurrent_dropout'].rvs() / 100
            name = str(dimension) + '_' + str(dropout) + '_' + str(recurrent_dropout)
            model_folder = self.folder + '/' + name
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            for choice in attention_choices:
                attention_name = choice
                for sentence_input in self.input_config:
                    input_name = 'word'
                    if sentence_input == [WORD, POS_TAG]:
                        input_name = 'word_pos'
                    if sentence_input == [WORD, POS_TAG, OFFSET]:
                        input_name = 'word_pos_offset'
                    config_folder = model_folder + '/' + input_name + attention_name
                    if not os.path.exists(config_folder):
                        os.mkdir(config_folder)
                    x = training_set.input_dict(sentence_input, list())
                    y = training_labels
                    val_x = validation_set.input_dict(sentence_input, list())
                    val_y = validation_labels
                    test_x = test_set.input_dict(sentence_input, list())
                    model = self.attention_model(sentence_input, layers, dimension, dropout, recurrent_dropout, choice)
                    class_weights = {0: 0.5, 1: 5, 2: 5, 3: 7, 4: 10}

                    history = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=128, verbose=2,
                                        class_weight=class_weights)
                    y_prob = model.predict(test_x)
                    y_classes = y_prob.argmax(axis=-1)
                    total_predictions = np.concatenate((negative_classes, y_classes))
                    print_tsv(config_folder, name, total_instances, total_predictions)
                    print_text_version(config_folder, name, total_instances, total_classes, total_predictions)
                    plot(config_folder, name, history)
                    f = open(config_folder + '/' + 'stats_' + name + '.txt', 'w')



    def configuration_search(self, training_set, training_labels, validation_set, validation_labels,
                             test_set, negative_classes, total_instances, total_classes, model_number: int):
        layers = 1
        epochs = 40
        attention_configurations = [(False, True)]
        for i in range(model_number):
            p = self.hyperparameters_dist
            dimension = p['dimension'].rvs() * 20
            dropout = p['dropout'].rvs() / 100
            recurrent_dropout = p['recurrent_dropout'].rvs() / 100
            name = str(dimension) + '_' + str(dropout) + '_' + str(recurrent_dropout)
            model_folder = self.folder + '/' + name
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            for (word_att, att) in attention_configurations:
                self.word_attention = word_att
                self.attention = att
                attention_name = ''
                if word_att and att:
                    attention_name = '+watt_att'
                if word_att and not att:
                    attention_name = '+watt'
                if not word_att and att:
                    attention_name = '+att'
                for sentence_input in self.input_config:
                    input_name = 'word'
                    if sentence_input == [WORD, POS_TAG]:
                        input_name = 'word_pos'
                    if sentence_input == [WORD, POS_TAG, OFFSET]:
                        input_name = 'word_pos_offset'
                    config_folder = model_folder + '/' + input_name + attention_name
                    if not os.path.exists(config_folder):
                        os.mkdir(config_folder)
                    x = training_set.input_dict(sentence_input, list())
                    y = training_labels
                    val_x = validation_set.input_dict(sentence_input, list())
                    val_y = validation_labels
                    test_x = test_set.input_dict(sentence_input, list())
                    model = self.create_model(sentence_input, layers, dimension, dropout, recurrent_dropout)
                    class_weights = {0: 0.5, 1: 5, 2: 5, 3: 7, 4: 10}

                    history = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=128, verbose=2,
                                        class_weight=class_weights)
                    y_prob = model.predict(test_x)
                    y_classes = y_prob.argmax(axis=-1)
                    total_predictions = np.concatenate((negative_classes, y_classes))
                    print_tsv(config_folder, name, total_instances, total_predictions)
                    print_text_version(config_folder, name, total_instances, total_classes, total_predictions)
                    plot(config_folder, name, history)
                    f = open(config_folder + '/' + 'stats_' + name + '.txt', 'w')
                    matrix, report = metrics(total_classes, total_predictions)
                    f.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(report, matrix))
                    f.close()

    def weight_search(self, training_set, training_labels, validation_set, validation_labels,
                             test_set, negative_classes, total_instances, total_classes, model_number: int):
        layers = 1
        epochs = 40
        p = self.hyperparameters_dist
        dimension = p['dimension']
        dropout = p['dropout']
        recurrent_dropout = p['recurrent_dropout']
        self.attention = True
        self.word_attention = False
        name = str(dimension) + '_' + str(dropout) + '_' + str(recurrent_dropout)
        sentence_input = [WORD, POS_TAG, OFFSET]
        input_name = 'word_pos_offset'
        model_folder = self.folder + '/' + name + '_' + input_name + '_att'
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        x = training_set.input_dict(sentence_input, list())
        y = training_labels
        val_x = validation_set.input_dict(sentence_input, list())
        val_y = validation_labels
        test_x = test_set.input_dict(sentence_input, list())
        model = self.create_model(sentence_input, layers, dimension, dropout, recurrent_dropout)
        weight_0 = stats.randint(1, 10)
        weight_1 = stats.randint(1, 4)
        weight_2 = stats.randint(4, 7)
        weight_3 = stats.randint(5, 9)
        weight_4 = stats.randint(9, 12)
        for i in range(model_number):
            class_weights = {0: weight_0.rvs()/10, 1: weight_1.rvs(), 2: weight_2.rvs(),
                             3: weight_3.rvs(), 4: weight_4.rvs()}
            weight_folder = model_folder + '/' + str(class_weights[0]) + '_' + str(class_weights[1]) + '_' + \
                            str(class_weights[2]) + '_' + str(class_weights[3]) + '_' + str(class_weights[4])
            if not os.path.exists(weight_folder):
                os.mkdir(weight_folder)
            history = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=128, verbose=2,
                                class_weight=class_weights)
            y_prob = model.predict(test_x)
            y_classes = y_prob.argmax(axis=-1)
            total_predictions = np.concatenate((negative_classes, y_classes))
            print_tsv(weight_folder, name, total_instances, total_predictions)
            print_text_version(weight_folder, name, total_instances, total_classes, total_predictions)
            plot(weight_folder, name, history)
            f = open(weight_folder + '/' + 'stats_' + name + '.txt', 'w')
            matrix, report = metrics(total_classes, total_predictions)
            f.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(report, matrix))
            f.close()

    def attention_model(self, config, layers, dimension, dropout, recurrent_dropout, choice: str) -> Model:
        final_input = list()
        complete_input = list()
        if WORD in config:
            word_input = Input(shape=(self.sentence_max_len, self.word_model.vector_size), name=SENTENCE_WORD)
            complete_input.append(word_input)
            final_input.append(word_input)
        if POS_TAG in config:
            tag_input = Input(shape=(self.sentence_max_len, self.tag_model.vector_size), name=SENTENCE_TAG)
            complete_input.append(tag_input)
            final_input.append(tag_input)
        if OFFSET in config:
            off1_input = Input(shape=(self.sentence_max_len,), name=SENTENCE_OFFSET1)
            off2_input = Input(shape=(self.sentence_max_len,), name=SENTENCE_OFFSET2)
            complete_input.append(off1_input)
            complete_input.append(off2_input)
            offset1_embedding = Embedding(output_dim=10, input_dim=self.off_dim, trainable=True)(off1_input)
            offset2_embedding = Embedding(output_dim=10, input_dim=self.off_dim, trainable=True)(off2_input)
            final_input.append(offset1_embedding)
            final_input.append(offset2_embedding)
        if len(final_input) == 1:
            lstm_input = final_input[0]
        else:
            lstm_input = concatenate(final_input, name='final_input')
        sentence_out = None
        if layers == 1:
            seq_sentence = Bidirectional(LSTM(dimension,
                                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                              dropout=dropout, return_sequences=True, return_state=False,
                                              recurrent_dropout=recurrent_dropout))(lstm_input)
            if choice == 'self_att':
                self_att = SeqSelfAttention()(seq_sentence)
                sentence_out = Flatten()(self_att)
            elif choice == 'context_att':
                sentence_out = AttentionWithContext()(seq_sentence)
            else:
                sentence_out = AttentionL(self.sentence_max_len)(seq_sentence)
        if layers > 1:
            layer_list = list()
            layer_list.append(lstm_input)
            for i in range(layers - 1):
                layer = Bidirectional(LSTM(dimension,
                                           dropout=dropout,
                                           return_sequences=True,
                                           recurrent_dropout=recurrent_dropout))(layer_list[i])
                layer_list.append(layer)
            last_sentence = Bidirectional(
                    LSTM(dimension,
                         dropout=dropout, return_sequences=True,
                         recurrent_dropout=recurrent_dropout))(layer_list[len(layer_list) - 1])
            if choice == 'self_att':
                self_att = SeqSelfAttention()(last_sentence)
                sentence_out = Flatten()(self_att)
            elif choice == 'context_att':
                sentence_out = AttentionWithContext()(last_sentence)
            else:
                sentence_out = AttentionL(self.sentence_max_len)(last_sentence)
        fully_connect = Dense(40, activation='relu', name='fc')(sentence_out)
        main_output = Dense(5, activation='softmax', name='main_output')(fully_connect)
        model = Model(inputs=complete_input, outputs=[main_output])
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.0001, decay=0, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])
        model.summary()
        return model

    def create_model(self, config, layers, dimension, dropout, recurrent_dropout) -> Model:
        candidate_1 = self.word_model.wv.get_vector('PairDrug1')
        candidate_1 = candidate_1.reshape(1, len(candidate_1))
        candidate_2 = self.word_model.wv.get_vector('PairDrug2')
        candidate_2 = candidate_2.reshape(1, len(candidate_2))
        final_input = list()
        complete_input = list()
        if WORD in config:
            word_input = Input(shape=(self.sentence_max_len, self.word_model.vector_size), name=SENTENCE_WORD)
            complete_input.append(word_input)
            if self.word_attention:
                candidate_input_1 = CandidateAttention(candidate_1)(word_input)
                candidate_input_2 = CandidateAttention(candidate_2)(word_input)
                candidate_input = average([candidate_input_1, candidate_input_2])
                final_input.append(candidate_input)
            else:
                final_input.append(word_input)
        if POS_TAG in config:
            tag_input = Input(shape=(self.sentence_max_len, self.tag_model.vector_size), name=SENTENCE_TAG)
            complete_input.append(tag_input)
            final_input.append(tag_input)
        if OFFSET in config:
            off1_input = Input(shape=(self.sentence_max_len,), name=SENTENCE_OFFSET1)
            off2_input = Input(shape=(self.sentence_max_len,), name=SENTENCE_OFFSET2)
            complete_input.append(off1_input)
            complete_input.append(off2_input)
            offset1_embedding = Embedding(output_dim=10, input_dim=self.off_dim, trainable=True)(off1_input)
            offset2_embedding = Embedding(output_dim=10, input_dim=self.off_dim, trainable=True)(off2_input)
            final_input.append(offset1_embedding)
            final_input.append(offset2_embedding)
        if len(final_input) == 1:
            lstm_input = final_input[0]
        else:
            lstm_input = concatenate(final_input, name='final_input')
        sentence_out = None
        if layers == 1:
            if self.attention:
                seq_sentence = Bidirectional(LSTM(dimension,
                                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                                  dropout=dropout, return_sequences=True, return_state=False,
                                                  recurrent_dropout=recurrent_dropout))(lstm_input)
                sentence_out = Attention()(seq_sentence)
            else:
                sentence_out = Bidirectional(LSTM(dimension,
                                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                                  dropout=dropout, return_sequences=False, return_state=False,
                                                  recurrent_dropout=recurrent_dropout))(lstm_input)
        if layers > 1:
            layer_list = list()
            layer_list.append(lstm_input)
            for i in range(layers-1):
                layer = Bidirectional(LSTM(dimension,
                                           kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                           dropout=dropout,
                                           return_sequences=True,
                                           recurrent_dropout=recurrent_dropout))(layer_list[i])
                layer_list.append(layer)
            if self.attention:
                last_sentence = Bidirectional(LSTM(dimension, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                              dropout=dropout, return_sequences=True,
                                              recurrent_dropout=recurrent_dropout))(layer_list[len(layer_list)-1])
                sentence_out = Attention()(last_sentence)
            else:
                sentence_out = Bidirectional(LSTM(dimension, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                              dropout=dropout, return_sequences=False,
                                              recurrent_dropout=recurrent_dropout))(layer_list[len(layer_list)-1])
        fully_connect = Dense(40, activation='relu', name='fc')(sentence_out)
        main_output = Dense(5, activation='softmax', name='main_output')(fully_connect)
        model = Model(inputs=complete_input, outputs=[main_output])
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.0001, decay=0, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])
        model.summary()
        return model
