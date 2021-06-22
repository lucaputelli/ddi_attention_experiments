from gensim.models import Word2Vec
from keras.layers import concatenate, Dense, LSTM, Bidirectional, Input, Embedding, Flatten, add, GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, History
from keras.initializers import *
from keras.optimizers import *
from keras_self_attention import SeqSelfAttention
from typing import Iterable, Tuple
from post_processing_lib import *
import os
from input_strings import *
from input_configurations import *


class DDIModel:

    def __init__(self, folder: str, sentence_max_len: int, path_max_len, off_dim: int, word_model: Word2Vec,
                 tag_model: Word2Vec, edge_model: Word2Vec, hyperparameters_dist: dict,
                 sentence_input=None, path_input=None, input_config=DEPENDENCY_PATH):
        self.folder = folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        # Default input configuration: WORD+POS+OFFSET, WORD+POS+EDGE
        (self.sentence_input, self.path_input) = complete
        if sentence_input is not None:
            self.sentence_input = sentence_input
        if path_input is not None:
            self.path_input = path_input
        self.sentence_max_len = sentence_max_len
        self.path_max_len = path_max_len
        self.off_dim = off_dim
        self.word_model = word_model
        self.tag_model = tag_model
        self.edge_model = edge_model
        self.histories = list()
        self.hyperparameters_dist = hyperparameters_dist
        self.hyperparameters_values = {}
        self.config_folder = self.folder
        self.checkpoint = None
        self.input_config = input_config
        self.attention = True
        self.pos_embedding = pos_embedding

    def create_model(self, layers, dimension, path_dimension, dropout, recurrent_dropout):
        sentence_input_for_concatenating = list()
        complete_input = list()
        if 'word' in self.sentence_input:
            sentence_word_input = Input(shape=(self.sentence_max_len, self.word_model.vector_size), name=SENTENCE_WORD)
            if 'offset' not in self.sentence_input or not self.pos_embedding:
                sentence_input_for_concatenating.append(sentence_word_input)
            complete_input.append(sentence_word_input)
            if 'offset' in self.sentence_input:
                if not self.pos_embedding:
                    sentence_off1_input = Input(shape=(self.sentence_max_len,), name=SENTENCE_OFFSET1)
                    sentence_off2_input = Input(shape=(self.sentence_max_len,), name=SENTENCE_OFFSET2)
                    offset1_embedding = Embedding(output_dim=3, input_dim=self.off_dim, trainable=True)(
                        sentence_off1_input)
                    offset2_embedding = Embedding(output_dim=3, input_dim=self.off_dim, trainable=True)(
                        sentence_off2_input)
                    sentence_input_for_concatenating.append(offset1_embedding)
                    sentence_input_for_concatenating.append(offset2_embedding)
                    complete_input.append(sentence_off1_input)
                    complete_input.append(sentence_off2_input)
                else:
                    sentence_off1_input = Input(shape=(self.sentence_max_len, self.word_model.vector_size),
                                                name=SENTENCE_OFFSET1)
                    sentence_off2_input = Input(shape=(self.sentence_max_len, self.word_model.vector_size),
                                                name=SENTENCE_OFFSET2)
                    complete_input.append(sentence_off1_input)
                    complete_input.append(sentence_off2_input)
                    word_off = add([sentence_word_input, sentence_off1_input, sentence_off2_input], name='word_off_sum')
                    sentence_input_for_concatenating.insert(0, word_off)
        if 'tag' in self.sentence_input:
            sentence_tag_input = Input(shape=(self.sentence_max_len, self.tag_model.vector_size), name=SENTENCE_TAG)
            sentence_input_for_concatenating.append(sentence_tag_input)
            complete_input.append(sentence_tag_input)
        if len(sentence_input_for_concatenating) == 1:
            final_input = sentence_input_for_concatenating[0]
        else:
            final_input = concatenate(sentence_input_for_concatenating, name='final_input')
        if layers == 1:
            if self.attention:
                seq_sentence = Bidirectional(LSTM(dimension,
                                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                              dropout=dropout, return_sequences=True, return_state=False,
                                              recurrent_dropout=recurrent_dropout))(final_input)
                att_sentence = SeqSelfAttention(attention_activation='softmax')(seq_sentence)
                flat_sentence = Flatten()(att_sentence)
            else:
                flat_sentence = Bidirectional(LSTM(dimension,
                                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                              dropout=dropout, return_sequences=False, return_state=False,
                                              recurrent_dropout=recurrent_dropout))(final_input)
        if layers > 1:
            layer_list = list()
            layer_list.append(final_input)
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
            # flat_sentence = AttentionL(self.sentence_max_len)(last_sentence)
                att_sentence = SeqSelfAttention(attention_activation='softmax')(last_sentence)
                flat_sentence = GlobalMaxPooling1D()(att_sentence)
                # flat_sentence = Flatten()(att_sentence)
            else:
                flat_sentence = Bidirectional(LSTM(dimension, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                              dropout=dropout, return_sequences=False,
                                              recurrent_dropout=recurrent_dropout))(layer_list[len(layer_list)-1])
        if len(self.path_input) != 0:
            if self.input_config == DEPENDENCY_PATH:
                path_input_layers = list()
                if 'word' in self.path_input:
                    path_word_input = Input(shape=(self.path_max_len, self.word_model.vector_size), name=PATH_WORD)
                    path_input_layers.append(path_word_input)
                    complete_input.append(path_word_input)
                if 'tag' in self.path_input:
                    path_tag_input = Input(shape=(self.path_max_len, self.tag_model.vector_size), name=PATH_TAG)
                    path_input_layers.append(path_tag_input)
                    complete_input.append(path_tag_input)
                if 'edge' in self.path_input:
                    path_edge_input = Input(shape=(self.path_max_len, self.edge_model.vector_size), name=PATH_EDGE)
                    path_input_layers.append(path_edge_input)
                    complete_input.append(path_edge_input)
                if len(path_input_layers) == 1:
                    path_final_input = path_input_layers[0]
                else:
                    path_final_input = concatenate(path_input_layers, name='path_final_input')
                seq_path = Bidirectional(LSTM(path_dimension,
                                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                              dropout=dropout,
                                              recurrent_dropout=recurrent_dropout))(path_final_input)
                lstm_output = concatenate([flat_sentence, seq_path])
            if self.input_config == VISITS:
                depth_input_layers = list()
                breadth_input_layers = list()
                if 'word' in self.path_input:
                    depth_word_input = Input(shape=(self.path_max_len, self.word_model.vector_size), name=DEPTH_WORD)
                    breadth_word_input = Input(shape=(self.path_max_len, self.word_model.vector_size), name=BREADTH_WORD)
                    depth_input_layers.append(depth_word_input)
                    breadth_input_layers.append(breadth_word_input)
                    complete_input.append(depth_word_input)
                    complete_input.append(breadth_word_input)
                if 'tag' in self.path_input:
                    depth_tag_input = Input(shape=(self.path_max_len, self.tag_model.vector_size), name=DEPTH_TAG)
                    breadth_tag_input = Input(shape=(self.path_max_len, self.tag_model.vector_size),
                                               name=BREADTH_TAG)
                    depth_input_layers.append(depth_tag_input)
                    breadth_input_layers.append(breadth_tag_input)
                    complete_input.append(depth_tag_input)
                    complete_input.append(breadth_tag_input)
                if 'edge' in self.path_input:
                    depth_edge_input = Input(shape=(self.path_max_len, self.edge_model.vector_size), name=DEPTH_EDGE)
                    breadth_edge_input = Input(shape=(self.path_max_len, self.edge_model.vector_size), name=BREADTH_EDGE)
                    depth_input_layers.append(depth_edge_input)
                    breadth_input_layers.append(breadth_edge_input)
                    complete_input.append(depth_edge_input)
                    complete_input.append(breadth_edge_input)
                if len(depth_input_layers) == 1:
                    depth_final_input = depth_input_layers[0]
                else:
                    depth_final_input = concatenate(depth_input_layers, name='depth_final_input')
                if len(breadth_input_layers) == 1:
                    breadth_final_input = breadth_input_layers[0]
                else:
                    breadth_final_input = concatenate(breadth_input_layers, name='breadth_final_input')
                if self.attention:
                    seq_depth = Bidirectional(LSTM(path_dimension,
                                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                               dropout=dropout, return_sequences=True,
                                               recurrent_dropout=recurrent_dropout))(depth_final_input)
                    att_depth = SeqSelfAttention(attention_activation='softmax')(seq_depth)
                    flat_depth = GlobalMaxPooling1D()(att_depth)
                    # flat_depth = Flatten()(att_depth)
                    seq_breadth = Bidirectional(LSTM(path_dimension,
                                                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                                 dropout=dropout, return_sequences=True,
                                                 recurrent_dropout=recurrent_dropout))(breadth_final_input)
                    att_breadth = SeqSelfAttention(attention_activation='softmax')(seq_breadth)
                    flat_breadth = GlobalMaxPooling1D()(att_breadth)
                    # flat_breadth = Flatten()(att_breadth)
                else:
                    flat_depth = Bidirectional(LSTM(path_dimension,
                                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                               dropout=dropout, return_sequences=False,
                                               recurrent_dropout=recurrent_dropout))(depth_final_input)
                    flat_breadth = Bidirectional(LSTM(path_dimension,
                                                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=1),
                                                 dropout=dropout, return_sequences=False,
                                                 recurrent_dropout=recurrent_dropout))(breadth_final_input)
                lstm_output = concatenate([flat_sentence, flat_depth, flat_breadth])
        else:
            lstm_output = flat_sentence
        main_output = Dense(5, activation='softmax', name='main_output')(lstm_output)
        name = str(dimension) + '_' + str(path_dimension) + '_' + str(dropout) + '_' + str(recurrent_dropout)
        model_folder = self.config_folder+'/'+name
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        self.checkpoint = ModelCheckpoint(model_folder + '/{epoch}_{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1,
                                          save_best_only=False,
                                          mode='auto')
        '''self.checkpoint = ModelCheckpoint(model_folder+'/pesi.hdf5', monitor='val_loss', verbose=1,
                                          save_best_only=False,
                                          mode='auto')'''
        model = Model(inputs=complete_input, outputs=[main_output])
        json_string = model.to_json()
        with open(model_folder+'/config.json', 'w') as config_file:
            config_file.write(json_string)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.001, decay=0, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])
        model.summary()
        return model

    def random_search(self, training_set, y, validation_set, val_y, n_iter: int) -> Iterable[Tuple[History, dict]]:
        for i in range(n_iter):
            x = training_set.input_dict(self.sentence_input, self.path_input)
            val_x = validation_set.input_dict(self.sentence_input, self.path_input)
            layers = self.hyperparameters_dist['sentence_layers'].rvs()
            dimension = self.hyperparameters_dist['dimension'].rvs()*20
            path_dimension = self.hyperparameters_dist['path_dimension'].rvs()*10
            dropout = self.hyperparameters_dist['dropout'].rvs()/100
            recurrent_dropout = self.hyperparameters_dist['recurrent_dropout'].rvs()/100
            epochs = self.hyperparameters_dist['epochs'].rvs()*2
            configuration = {'dimension': dimension, 'path_dimension': path_dimension,
                             'dropout': dropout, 'recurrent_dropout': recurrent_dropout,
                             'epochs': epochs}
            model = self.create_model(layers, dimension, path_dimension, dropout, recurrent_dropout)
            history = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=128, verbose=2, callbacks=[self.checkpoint])
            self.histories.append((history, configuration, model))
            weights = model.get_weights()
        return self.histories

    def grid_search(self, x, y, val_x, val_y, param_values: dict) -> Iterable[Tuple[History, dict]]:
        self.histories = list()
        self.hyperparameters_values = param_values
        dimensions = self.hyperparameters_values['dimension']
        path_dimensions = self.hyperparameters_values['path_dimension']
        dropouts = self.hyperparameters_values['dropout']
        recurrent_dropouts = self.hyperparameters_values['recurrent_dropout']
        combinations = [(dim, p_dim, drop, r_drop) for dim in dimensions for p_dim in path_dimensions
                        for drop in dropouts for r_drop in recurrent_dropouts]
        for (dim, p_dim, drop, r_drop) in combinations:
            model = self.create_model(dim, p_dim, drop, r_drop)
            history = model.fit(x, y, validation_data=(val_x, val_y), epochs=10, batch_size=128, verbose=2)
            self.histories.append(history)
        return self.histories

    def best_model(self):
        max_acc = 0
        if len(self.histories) == 0:
            raise Exception('No models present')
        else:
            for (history, configuration, model) in self.histories:
                accuracy = history.history['acc']
                last_acc = accuracy[len(accuracy)-1]
                if last_acc > max_acc:
                    max_acc = last_acc
                    max_configuration = configuration
        return max_acc, max_configuration

    def testing(self, test_sentence, test_path, test_offset1, test_offset2, negative_classes, total_instances, total_classes):
        for (history, configuration, model) in self.histories:
            name = str(configuration['dimension'])+'_'+str(configuration['path_dimension'])+'_' + \
                   str(configuration['dropout'])+'_'+str(configuration['recurrent_dropout'])
            model_folder = self.folder + '/' + name
            y_prob = model.predict([test_sentence, test_path, test_offset1, test_offset2])
            y_classes = y_prob.argmax(axis=-1)
            total_predictions = np.concatenate((negative_classes, y_classes))
            print_tsv(model_folder, name, total_instances, total_predictions)
            print_text_version(model_folder, name, total_instances, total_classes, total_predictions)
            plot(model_folder, name, history)
            f = open(model_folder+'/'+'stats_'+name+'.txt', 'w')
            matrix, report = metrics(total_classes, total_predictions)
            f.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(report, matrix))
            f.close()

    def random_configuration_search(self, training_set, y, validation_set, val_y, test_set, negative_classes,
                                    total_instances, total_classes, input_configurations, iterations: int):
        for i in range(iterations):
            layers = self.hyperparameters_dist['sentence_layers'].rvs()
            dimension = self.hyperparameters_dist['dimension'].rvs() * 20
            path_dimension = self.hyperparameters_dist['path_dimension'].rvs() * 10
            dropout = self.hyperparameters_dist['dropout'].rvs() / 100
            recurrent_dropout = self.hyperparameters_dist['recurrent_dropout'].rvs() / 100
            epochs = self.hyperparameters_dist['epochs'].rvs() * 2
            configuration = {'layers': layers, 'dimension': dimension, 'path_dimension': path_dimension,
                             'dropout': dropout, 'recurrent_dropout': recurrent_dropout,
                             'epochs': epochs}
            print("Testing", configuration)
            for (sentence_input, path_input) in input_configurations:
                self.sentence_input = sentence_input
                self.path_input = path_input
                s_name = ''
                for i in range(len(self.sentence_input)):
                    inp = self.sentence_input[i]
                    if i == len(self.sentence_input)-1:
                        s_name = s_name + inp
                    else:
                        s_name = s_name + inp + '_'
                p_name = ''
                for i in range(len(self.path_input)):
                    inp = self.path_input[i]
                    if i == len(self.path_input)-1:
                        p_name = p_name + inp
                    else:
                        p_name = p_name + inp + '_'
                print(s_name, p_name)
                print('With:', sentence_input, path_input)
                if p_name == '':
                    self.config_folder = self.folder + '/'+s_name
                else:
                    self.config_folder = self.folder + '/' + s_name + '-' + p_name
                if not os.path.exists(self.config_folder):
                    os.mkdir(self.config_folder)
                if self.input_config == VISITS:
                    x = training_set.visits_dict(sentence_input, path_input)
                    val_x = validation_set.visits_dict(sentence_input, path_input)
                    test_x = test_set.visits_dict(sentence_input, path_input)
                else:
                    x = training_set.input_dict(sentence_input, path_input)
                    val_x = validation_set.input_dict(sentence_input, path_input)
                    test_x = test_set.input_dict(sentence_input, path_input)
                model = self.create_model(layers, dimension, path_dimension, dropout, recurrent_dropout)
                history = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=128, verbose=2,
                                    callbacks=[self.checkpoint])
                self.histories.append((history, configuration, model))
                name = str(configuration['dimension']) + '_' + str(configuration['path_dimension']) + '_' + \
                       str(configuration['dropout']) + '_' + str(configuration['recurrent_dropout'])
                name = str(dimension) + '_' + str(path_dimension) + '_' + str(dropout) + '_' + str(recurrent_dropout)
                model_folder = self.config_folder + '/' + name
                y_prob = model.predict(test_x)
                y_classes = y_prob.argmax(axis=-1)
                total_predictions = np.concatenate((negative_classes, y_classes))
                print_tsv(model_folder, name, total_instances, total_predictions)
                print_text_version(model_folder, name, total_instances, total_classes, total_predictions)
                plot(model_folder, name, history)
                f = open(model_folder + '/' + 'stats_' + name + '.txt', 'w')
                matrix, report = metrics(total_classes, total_predictions)
                f.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(report, matrix))
                f.close()
        return self.histories

    def random_attention(self, training_set, y, validation_set, val_y, test_set, negative_classes,
                         total_instances, total_classes, input_configurations, iterations: int):
        for i in range(iterations):
            layers = self.hyperparameters_dist['sentence_layers'].rvs()
            dimension = self.hyperparameters_dist['dimension'].rvs() * 20
            path_dimension = self.hyperparameters_dist['path_dimension'].rvs() * 10
            dropout = self.hyperparameters_dist['dropout'].rvs() / 100
            recurrent_dropout = self.hyperparameters_dist['recurrent_dropout'].rvs() / 100
            epochs = self.hyperparameters_dist['epochs'].rvs() * 2
            configuration = {'layers': layers, 'dimension': dimension, 'path_dimension': path_dimension,
                             'dropout': dropout, 'recurrent_dropout': recurrent_dropout,
                             'epochs': epochs}
            print("Testing", configuration)
            for (sentence_input, path_input) in input_configurations:
                self.sentence_input = sentence_input
                self.path_input = path_input
                s_name = ''
                for i in range(len(self.sentence_input)):
                    inp = self.sentence_input[i]
                    if i == len(self.sentence_input)-1:
                        s_name = s_name + inp
                    else:
                        s_name = s_name + inp + '_'
                p_name = ''
                for i in range(len(self.path_input)):
                    inp = self.path_input[i]
                    if i == len(self.path_input)-1:
                        p_name = p_name + inp
                    else:
                        p_name = p_name + inp + '_'
                print(s_name, p_name)
                print('With:', sentence_input, path_input)
                if p_name == '':
                    self.config_folder = self.folder + '/'+s_name
                else:
                    self.config_folder = self.folder + '/' + s_name + '-' + p_name
                if not os.path.exists(self.config_folder):
                    os.mkdir(self.config_folder)
                if self.input_config == VISITS:
                    x = training_set.visits_dict(sentence_input, path_input)
                    val_x = validation_set.visits_dict(sentence_input, path_input)
                    test_x = test_set.visits_dict(sentence_input, path_input)
                else:
                    x = training_set.input_dict(sentence_input, path_input)
                    val_x = validation_set.input_dict(sentence_input, path_input)
                    test_x = test_set.input_dict(sentence_input, path_input)
                attention = [True, False]
                for b in attention:
                    self.attention = b
                    if b:
                        print('PRESENT ATTENTION LAYER')
                    else:
                        print('WITHOUT ATTENTION LAYER')
                    model = self.create_model(layers, dimension, path_dimension, dropout, recurrent_dropout)
                    history = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=128, verbose=2,
                                        callbacks=[self.checkpoint])
                    self.histories.append((history, configuration, model))
                    name = str(configuration['dimension']) + '_' + str(configuration['path_dimension']) + '_' + \
                           str(configuration['dropout']) + '_' + str(configuration['recurrent_dropout'])
                    model_folder = self.config_folder + '/' + name
                    y_prob = model.predict(test_x)
                    y_classes = y_prob.argmax(axis=-1)
                    total_predictions = np.concatenate((negative_classes, y_classes))
                    print_tsv(model_folder, name, total_instances, total_predictions)
                    print_text_version(model_folder, name, total_instances, total_classes, total_predictions)
                    plot(model_folder, name, history)
                    f = open(model_folder + '/' + 'stats_' + name + '.txt', 'w')
                    matrix, report = metrics(total_classes, total_predictions)
                    f.write('Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(report, matrix))
                    # metrics_dump = open(model_folder+'/stats_dump_'+name+'.pkl', 'wb')
                    # pickle.dump((matrix, report), metrics_dump)
                    # (dump_matrix, dump_report) = pickle.load(metrics_dump)
                    f.close()
        return self.histories
