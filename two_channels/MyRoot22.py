import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
import sklearn
from pre_processing_lib import *
import tensorflow
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam, Adadelta
from AttentionMechanism import *
import pickle

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib;

matplotlib.use('agg')
from matplotlib import pyplot


def metrics(t_labels, t_predictions):
    numeric_labels = np.argmax(t_labels, axis=1)
    target_names = ['unrelated', 'effect', 'mechanism', 'advise', 'int']
    matrix = confusion_matrix(numeric_labels, t_predictions)
    FP = (matrix.sum(axis=0) - np.diag(matrix))[1:]
    FN = (matrix.sum(axis=1) - np.diag(matrix))[1:]
    TP = (np.diag(matrix))[1:]
    overall_fp = np.sum(FP)
    overall_fn = np.sum(FN)
    overall_tp = np.sum(TP)
    overall_precision = overall_tp / (overall_tp + overall_fp)
    overall_recall = overall_tp / (overall_tp + overall_fn)
    overall_f_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    report = classification_report(numeric_labels, t_predictions, target_names=target_names)
    return matrix, report, overall_precision, overall_recall, overall_f_score


def plot(folder: str, name: str, history):
    pyplot.clf()
    pyplot.figure(1, figsize=(13, 6))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.subplot(1, 2, 2)
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('model train vs validation accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.tight_layout()
    pyplot.savefig(folder + '/' + name + '.png')


def network1(dim1, dim2, dim3, lstm_units, dropout=0.2, r_dropout=0.2):
    # Inputs
    input1 = Input(
        shape=(dim1, 204))  # 204 perchè oltre al 200 abbiamo le 4 caratteristiche della parola (es. articolo)
    input2 = Input(shape=(dim2, 204))
    input3 = Input(shape=(dim3, 204))

    # LSTM nets
    lstm1 = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(input1)
    lstm2 = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(input2)
    lstm3 = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(input3)

    concat = Concatenate(axis=1)([lstm1, lstm2, lstm3])
    att = AttentionL(dim1 + dim2 + dim3)(concat)

    classification = Dense(5, activation='softmax')(att)
    model = Model(inputs=[input1, input2, input3], outputs=classification)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(),
                  metrics=['accuracy'])

    return model

def network2Canali(dim1, dim2, lstm_units, dropout=0.2, r_dropout=0.2):
    # Inputs
    input1 = Input(shape=(dim1, 204))
    input2 = Input(shape=(dim2, 204))

    # LSTM nets
    lstm1 = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(input1)
    lstm2 = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(input2)

    concat = Concatenate(axis=1)([lstm1, lstm2])
    att = AttentionL(dim1 + dim2)(concat)

    classification = Dense(5, activation='softmax')(att)
    model = Model(inputs=[input1, input2], outputs=classification)
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999),
    #              metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(),
                  metrics=['accuracy'])
    return model

def network2(dim, lstm_units, dropout=0.2, r_dropout=0.2):
    # Inputs
    inputlayer = Input(
        shape=(dim, 204))  # 204 perchè oltre al 200 abbiamo le 4 caratteristiche della parola (es. articolo)

    # LSTM nets
    lstm = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(inputlayer)

    att = AttentionL(dim)(lstm)

    classification = Dense(5, activation='softmax')(att)
    model = Model(inputs=[inputlayer], outputs=classification)
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999),
    #              metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(),
                  metrics=['accuracy'])
    return model


def network_1_85(lstm_units, dropout=0.2, r_dropout=0.2):
    # Inputs
    inputlayer = Input(
        shape=(85, 204))  # 204 perchè oltre al 200 abbiamo le 4 caratteristiche della parola (es. articolo)

    # LSTM nets
    lstm = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(inputlayer)
    att = AttentionL(85)(lstm)

    classification = Dense(5, activation='softmax')(att)
    model = Model(inputs=[inputlayer], outputs=classification)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    return model

def network_1_85_doppioLSTM(lstm_units, dropout=0.2, r_dropout=0.2):
    # Inputs
    inputlayer = Input(
        shape=(85, 204))  # 204 perchè oltre al 200 abbiamo le 4 caratteristiche della parola (es. articolo)

    # LSTM nets
    lstm = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(inputlayer)
    lstm1 = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(lstm)
    att = AttentionL(85)(lstm1)

    classification = Dense(5, activation='softmax')(att)
    model = Model(inputs=[inputlayer], outputs=classification)
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
    #              metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(),
                  metrics=['accuracy'])
    return model

def network_1_95(lstm_units, dropout=0.2, r_dropout=0.2):
    # Inputs
    inputlayer = Input(
        shape=(95, 204))  # 204 perchè oltre al 200 abbiamo le 4 caratteristiche della parola (es. articolo)

    # LSTM nets
    lstm = LSTM(lstm_units, dropout=dropout, recurrent_dropout=r_dropout, return_sequences=True)(inputlayer)
    att = AttentionL(95)(lstm)

    classification = Dense(5, activation='softmax')(att)
    model = Model(inputs=[inputlayer], outputs=classification)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    return model

def preprocessing():
    sents = get_sentences('Dataset/Train/Overall')

    v_sents = get_sentences('Dataset/Train/Validation')
    test_sents = get_sentences('Dataset/Test/Overall')
    instances = get_instances(sents)
    instances = [x for x in instances if x is not None]
    instances = negative_filtering(instances)
    instances = [x for x in instances if x.get_dependency_path() is not None and len(x.get_dependency_path()) > 0]

    v_instances = get_instances(v_sents)
    v_instances = [x for x in v_instances if x is not None]
    v_instances = negative_filtering(v_instances)
    v_instances = [x for x in v_instances if x.get_dependency_path() is not None and len(x.get_dependency_path()) > 0]

    t_instances = get_instances(test_sents)
    t_selected = [x for x in t_instances if x is not None]
    t_negative = [x for x in t_instances if x is None]
    t_selected, t_negative2 = blind_negative_filtering(t_selected)
    t_negative.extend(t_negative2)
    t_negative.extend([x for x in t_selected if x.get_dependency_path() is None or len(x.get_dependency_path()) == 0])
    t_selected = [x for x in t_selected if x.get_dependency_path() is not None and len(x.get_dependency_path()) > 0]

    sents, labels = get_labelled_instances(instances)
    # return sents, labels
    v_sents, v_labels = get_labelled_instances(v_instances)
    t_sents, t_labels = get_labelled_instances(t_selected)
    return sents, labels, v_sents, v_labels, t_sents, t_labels


def preprocessing_prova():
    sents = get_sentences('Dataset/Train/Prova')
    instances = get_instances(sents)
    instances = [x for x in instances if x is not None]
    instances = negative_filtering(instances)
    instances = [x for x in instances if x.get_dependency_path() is not None and len(x.get_dependency_path()) > 0]
    sents, labels = get_labelled_instances(instances)
    return sents, labels


def verifica_Dimensione_divisione_frase(sents, val1, val2, val3):
    X_train = np.zeros((len(sents), 3, 3))
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    for i, x in enumerate(sents):
        # divisione della mia frase in 3
        c1 = 0
        c2 = 0
        c3 = 0
        val = 1
        for j in range(0, len(x)):
            if val == 1:
                c1 = c1 + 1
                if x[j].text == "PairDrug1":
                    X_train[i][0][2] = c1
                    if c1 >= val1:
                        lunghezza = 0
                        for k in range(0, c1):
                            if x[k].text != "Drug":
                                lunghezza = lunghezza + 1
                        X_train[i][0][0] = 1
                        X_train[i][0][1] = lunghezza
                        if lunghezza > val1:
                            X_train[i][0][0] = 2
                            lunghezza = 0
                            for k in range(0, c1):
                                if x[k].text != "Drug" and x[k].pos_ != 'DET':
                                    lunghezza = lunghezza + 1
                            X_train[i][0][1] = lunghezza
                            if lunghezza > val1:
                                X_train[i][0][0] = 3
                                lunghezza = 0
                                for k in range(0, c1):
                                    if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ':
                                        lunghezza = lunghezza + 1
                                X_train[i][0][1] = lunghezza
                                if lunghezza > val1:
                                    X_train[i][0][0] = 4
                                    lunghezza = 0
                                    for k in range(0, c1):
                                        if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[
                                            k].pos_ != 'SCONJ':
                                            lunghezza = lunghezza + 1
                                    X_train[i][0][1] = lunghezza
                                    if lunghezza > val1:
                                        X_train[i][0][0] = 5
                                        lunghezza = 0
                                        for k in range(0, c1):
                                            if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and \
                                                    x[k].pos_ != 'SCONJ' and x[k].pos_ != "PUNCT":
                                                lunghezza = lunghezza + 1
                                        X_train[i][0][1] = lunghezza
                                        if lunghezza > val1:
                                            X_train[i][0][0] = 6
                                            X_train[i][0][1] = lunghezza
                    else:
                        X_train[i][0][0] = 0
                        X_train[i][0][1] = c1
                    val = 2
            if val == 2:
                c2 = c2 + 1
                if x[j].text == "PairDrug2":
                    c2 = c2 - 1
                    c3 = c3 + 1
                    X_train[i][1][2] = c2
                    lunghezza = 0
                    if c2 >= val2:
                        for k in range(c1, c2):
                            if x[k].text != "Drug":
                                lunghezza = lunghezza + 1
                        X_train[i][1][0] = 1
                        X_train[i][1][1] = lunghezza
                        if lunghezza > val2:
                            X_train[i][1][0] = 2
                            lunghezza = 0
                            for k in range(c1, c2):
                                if x[k].text != "Drug" and x[k].pos_ != 'DET':
                                    lunghezza = lunghezza + 1
                            X_train[i][1][1] = lunghezza
                            if lunghezza > val2:
                                X_train[i][1][0] = 3
                                lunghezza = 0
                                for k in range(c1, c2):
                                    if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ':
                                        lunghezza = lunghezza + 1
                                X_train[i][1][1] = lunghezza
                                if lunghezza > val2:
                                    X_train[i][1][0] = 4
                                    lunghezza = 0
                                    for k in range(c1, c2):
                                        if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[
                                            k].pos_ != 'SCONJ':
                                            lunghezza = lunghezza + 1
                                    X_train[i][1][1] = lunghezza
                                    if lunghezza > val2:
                                        X_train[i][1][0] = 5
                                        lunghezza = 0
                                        for k in range(c1, c2):
                                            if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and \
                                                    x[k].pos_ != 'SCONJ' and x[k].pos_ != "PUNCT":
                                                lunghezza = lunghezza + 1
                                        X_train[i][1][1] = lunghezza
                                        if lunghezza > val2:
                                            X_train[i][1][0] = 6
                                            X_train[i][1][1] = lunghezza
                    else:
                        X_train[i][1][0] = 0
                        X_train[i][1][1] = c2
                    val = 3
            if val == 3:
                c3 = c3 + 1
        lunghezza = 0
        X_train[i][2][2] = c3
        if c3- c2 >= val3:
            for k in range(c2, c3):
                if x[k].text != "Drug":
                    lunghezza = lunghezza + 1
            X_train[i][2][0] = 1
            X_train[i][2][1] = lunghezza
            if lunghezza > val3:
                X_train[i][2][0] = 2
                lunghezza = 0
                for k in range(c2, c3):
                    if x[k].text != "Drug" and x[k].pos_ != 'DET':
                        lunghezza = lunghezza + 1
                X_train[i][2][1] = lunghezza
                if lunghezza > val3:
                    X_train[i][2][0] = 3
                    lunghezza = 0
                    for k in range(c2, c3):
                        if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ':
                            lunghezza = lunghezza + 1
                    X_train[i][2][1] = lunghezza
                    if lunghezza > val3:
                        X_train[i][2][0] = 4
                        lunghezza = 0
                        for k in range(c2, c3):
                            if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[
                                k].pos_ != 'SCONJ':
                                lunghezza = lunghezza + 1
                        X_train[i][2][1] = lunghezza
                        if lunghezza > val3:
                            X_train[i][2][0] = 5
                            lunghezza = 0
                            for k in range(c2, c3):
                                if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[
                                    k].pos_ != 'SCONJ' and x[k].pos_ != "PUNCT":
                                    lunghezza = lunghezza + 1
                            X_train[i][2][1] = lunghezza
                            if lunghezza > val3:
                                X_train[i][2][0] = 6
                                X_train[i][2][1] = lunghezza
        else:
            X_train[i][2][0] = 0
            X_train[i][2][1] = c3
        # TROVO IL MASSIMO DELLA LUNGHEZZA IN PAROLE DI OGNI PEZZO DI FRASE


    ##### ora ho quanto è lunga ogni singola frase e quanto dovro togliergli affinchè rispetti i canoni prescritti
    X_train1 = np.zeros((len(sents), val1, 204))
    X_train2 = np.zeros((len(sents), val2, 204))
    X_train3 = np.zeros((len(sents), val3, 204))

    for i, x in enumerate(sents):
        parte1, parte2, parte3 = ritorna_frase_pulita(word_model, word_model_4, x, i, X_train[i], val1, val2, val3)
        X_train1[i] = parte1
        X_train2[i] = parte2
        X_train3[i] = parte3
    return X_train1, X_train2, X_train3

#done
def verifica_Dimensione_divisione_frase_conDrug(sents, val1, val2, val3):
    X_train = np.zeros((len(sents), 3, 3))
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    for i, x in enumerate(sents):
        # divisione della mia frase in 3
        c1 = 0
        c2 = 0
        c3 = 0
        val = 1
        for j in range(0, len(x)):
            if val == 1:
                c1 = c1 + 1
                if x[j].text == "PairDrug1":
                    X_train[i][0][2] = c1
                    if c1 >= val1:
                        lunghezza = c1+1
                        #for k in range(0, c1):
                        #    if x[k].text != "Drug":
                        #        lunghezza = lunghezza + 1
                        #X_train[i][0][0] = 1
                        #X_train[i][0][1] = lunghezza
                        if lunghezza > val1:
                            X_train[i][0][0] = 2
                            lunghezza = 0
                            for k in range(0, c1):
                                if  x[k].pos_ != 'DET':
                                    lunghezza = lunghezza + 1
                            X_train[i][0][1] = lunghezza
                            if lunghezza > val1:
                                X_train[i][0][0] = 3
                                lunghezza = 0
                                for k in range(0, c1):
                                    if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ':
                                        lunghezza = lunghezza + 1
                                X_train[i][0][1] = lunghezza
                                if lunghezza > val1:
                                    X_train[i][0][0] = 4
                                    lunghezza = 0
                                    for k in range(0, c1):
                                        if x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[ \
                                            k].pos_ != 'SCONJ':
                                            lunghezza = lunghezza + 1
                                    X_train[i][0][1] = lunghezza
                                    if lunghezza > val1:
                                        X_train[i][0][0] = 5
                                        lunghezza = 0
                                        for k in range(0, c1):
                                            if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and \
                                                    x[k].pos_ != 'SCONJ' and x[k].pos_ != "PUNCT":
                                                lunghezza = lunghezza + 1
                                        X_train[i][0][1] = lunghezza
                                        if lunghezza > val1:
                                            X_train[i][0][0] = 6
                                            X_train[i][0][1] = lunghezza
                    else:
                        X_train[i][0][0] = 0
                        X_train[i][0][1] = c1
                    val = 2
            if val == 2:
                c2 = c2 + 1
                if x[j].text == "PairDrug2":
                    c2 = c2 - 1
                    c3 = c3 + 1
                    X_train[i][1][2] = c2
                    lunghezza = 0
                    if c2 >= val2:
                        lunghezza = c2+1
                        #for k in range(c1, c2):
                        #    if x[k].text != "Drug":
                        #        lunghezza = lunghezza + 1
                        #X_train[i][1][0] = 1
                        #X_train[i][1][1] = lunghezza
                        if lunghezza > val2:
                            X_train[i][1][0] = 2
                            lunghezza = 0
                            for k in range(c1, c2):
                                if x[k].pos_ != 'DET':
                                    lunghezza = lunghezza + 1
                            X_train[i][1][1] = lunghezza
                            if lunghezza > val2:
                                X_train[i][1][0] = 3
                                lunghezza = 0
                                for k in range(c1, c2):
                                    if x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ':
                                        lunghezza = lunghezza + 1
                                X_train[i][1][1] = lunghezza
                                if lunghezza > val2:
                                    X_train[i][1][0] = 4
                                    lunghezza = 0
                                    for k in range(c1, c2):
                                        if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[\
                                            k].pos_ != 'SCONJ':
                                            lunghezza = lunghezza + 1
                                    X_train[i][1][1] = lunghezza
                                    if lunghezza > val2:
                                        X_train[i][1][0] = 5
                                        lunghezza = 0
                                        for k in range(c1, c2):
                                            if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and \
                                                    x[k].pos_ != 'SCONJ' and x[k].pos_ != "PUNCT":
                                                lunghezza = lunghezza + 1
                                        X_train[i][1][1] = lunghezza
                                        if lunghezza > val2:
                                            X_train[i][1][0] = 6
                                            X_train[i][1][1] = lunghezza
                    else:
                        X_train[i][1][0] = 0
                        X_train[i][1][1] = c2
                    val = 3
            if val == 3:
                c3 = c3 + 1
        lunghezza = 0
        X_train[i][2][2] = c3
        if c3 >= val3:
            lunghezza = c3 + 1
            #for k in range(c2, c3):
            #    if x[k].text != "Drug":
            #        lunghezza = lunghezza + 1
            #X_train[i][2][0] = 1
            #X_train[i][2][1] = lunghezza
            if lunghezza > val3:
                X_train[i][2][0] = 2
                lunghezza = 0
                for k in range(c2, c3):
                    if  x[k].pos_ != 'DET':
                        lunghezza = lunghezza + 1
                X_train[i][2][1] = lunghezza
                if lunghezza > val3:
                    X_train[i][2][0] = 3
                    lunghezza = 0
                    for k in range(c2, c3):
                        if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ':
                            lunghezza = lunghezza + 1
                    X_train[i][2][1] = lunghezza
                    if lunghezza > val3:
                        X_train[i][2][0] = 4
                        lunghezza = 0
                        for k in range(c2, c3):
                            if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[\
                                k].pos_ != 'SCONJ':
                                lunghezza = lunghezza + 1
                        X_train[i][2][1] = lunghezza
                        if lunghezza > val3:
                            X_train[i][2][0] = 5
                            lunghezza = 0
                            for k in range(c2, c3):
                                if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[\
                                    k].pos_ != 'SCONJ' and x[k].pos_ != "PUNCT":
                                    lunghezza = lunghezza + 1
                            X_train[i][2][1] = lunghezza
                            if lunghezza > val3:
                                X_train[i][2][0] = 6
                                X_train[i][2][1] = lunghezza
        else:
            X_train[i][2][0] = 0
            X_train[i][2][1] = c3
        # TROVO IL MASSIMO DELLA LUNGHEZZA IN PAROLE DI OGNI PEZZO DI FRASE


    ##### ora ho quanto è lunga ogni singola frase e quanto dovro togliergli affinchè rispetti i canoni prescritti
    X_train1 = np.zeros((len(sents), val1, 204))
    X_train2 = np.zeros((len(sents), val2, 204))
    X_train3 = np.zeros((len(sents), val3, 204))

    for i, x in enumerate(sents):
        parte1, parte2, parte3 = ritorna_frase_pulita_conDrug(word_model, word_model_4, x, i, X_train[i], val1, val2, val3)
        X_train1[i] = parte1
        X_train2[i] = parte2
        X_train3[i] = parte3
    return X_train1, X_train2, X_train3

def ritorna_frase_pulita(word_model, word_model_4, x, i, frase, val1, val2, val3):
    primoPezzo = np.zeros((val1, 204))
    secondoPezzo = np.zeros((val2, 204))
    terzoPezzo = np.zeros((val3, 204))
    ###primo pezzo
    if int(frase[0][0]) == 0:
        indice = 0
        for i in range(0, int(frase[0][2])):
            try:
                primoPezzo[indice, :200] = word_model.wv[x[i].text]
                primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 1:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 2:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 3:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 4:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 5:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and x[
                    i].pos_ != "PUNCT":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 6:
        differenza = int(frase[0][1]) - val1 #quanti elementi sono in più nella mia frase
        rimozione = np.zeros(differenza)
        for i in range(0, len(rimozione)):
            randomNumber = np.random.randint(0, int(frase[0][2]) - 1)
            unico = True
            for j in range(0, i):
                if rimozione[j] == randomNumber or (
                        x[randomNumber].text != "Drug" and x[randomNumber].pos_ != "DET" and x[
                    randomNumber].pos_ != "CCONJ" and x[randomNumber].pos_ != "SCONJ" and \
                        x[randomNumber].pos_ != "PUNCT"):
                    j = i + 1
                    i = i - 1
                    unico = False
            if unico:
                rimozione[i] = randomNumber
        indice = 0
        for i in range(0, int(frase[0][2])):
            NonDaRimuovere = True
            for j in range(0, len(rimozione)):
                if i == rimozione[j]:
                    NonDaRimuovere = False
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and \
                        x[i].pos_ != "PUNCT" and NonDaRimuovere:
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass

    ###secondo pezzo di frase
    if int(frase[1][0]) == 0:
        indice = 0
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 1:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 2:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 3:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 4:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 5:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and x[
                    i].pos_ != "PUNCT":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 6:
        differenza = int(frase[1][1]) - val2
        rimozione = np.zeros(differenza)
        for i in range(0, len(rimozione)):
            randomNumber = np.random.randint(int(frase[0][2]) + 1, int(frase[1][2]))
            unico = True
            for j in range(0, i):
                if rimozione[j] == randomNumber or (
                        x[randomNumber].text != "Drug" and x[randomNumber].pos_ != "DET" and x[
                    randomNumber].pos_ != "CCONJ" and x[randomNumber].pos_ != "SCONJ" and \
                        x[randomNumber].pos_ != "PUNCT"):
                    j = i + 1
                    i = i - 1
                    unico = False
            if unico:
                rimozione[i] = randomNumber
        indice = 0
        for i in range(int(frase[1][2]), int(frase[2][2])):
            NonDaRimuovere = True
            for j in range(0, len(rimozione)):
                if i == rimozione[j]:
                    NonDaRimuovere = False
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and \
                        x[i].pos_ != "PUNCT" and NonDaRimuovere:
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass

    ###terzo pezzo di frase
    if int(frase[2][0]) == 0:
        indice = 0
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 1:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 2:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 3:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 4:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 5:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and x[
                    i].pos_ != "PUNCT":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 6:
        differenza = int(frase[2][1]) - val3
        rimozione = np.zeros(differenza)
        for i in range(0, len(rimozione)):
            randomNumber = np.random.randint(int(frase[1][2]) + 2, int(frase[2][2]))
            unico = True
            for j in range(0, i):
                if rimozione[j] == randomNumber or (
                        x[randomNumber].text != "Drug" and x[randomNumber].pos_ != "DET" and x[
                    randomNumber].pos_ != "CCONJ" and x[randomNumber].pos_ != "SCONJ" and \
                        x[randomNumber].pos_ != "PUNCT"):
                    j = i + 1
                    i = i - 1
                    unico = False
            if unico:
                rimozione[i] = randomNumber
        indice = 0
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            NonDaRimuovere = True
            for j in range(0, len(rimozione)):
                if i == rimozione[j]:
                    NonDaRimuovere = False
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and \
                        x[i].pos_ != "PUNCT" and NonDaRimuovere:
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    return primoPezzo, secondoPezzo, terzoPezzo

#done
def ritorna_frase_pulita_conDrug(word_model, word_model_4, x, i, frase, val1, val2, val3):
    primoPezzo = np.zeros((val1, 204))
    secondoPezzo = np.zeros((val2, 204))
    terzoPezzo = np.zeros((val3, 204))
    ###primo pezzo
    if int(frase[0][0]) == 0:
        indice = 0
        for i in range(0, int(frase[0][2])):
            try:
                primoPezzo[indice, :200] = word_model.wv[x[i].text]
                primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 1:
        indice = 0;
        '''
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
        '''
    elif int(frase[0][0]) == 2:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 3:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 4:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 5:
        indice = 0;
        for i in range(0, int(frase[0][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and x[
                    i].pos_ != "PUNCT":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0][0]) == 6:
        differenza = int(frase[0][1]) - val1 #quanti elementi sono in più nella mia frase
        rimozione = np.zeros(differenza)
        for i in range(0, len(rimozione)):
            randomNumber = np.random.randint(0, int(frase[0][2]) - 1)
            unico = True
            for j in range(0, i):
                if rimozione[j] == randomNumber or (
                        x[randomNumber].text != "Drug" and x[randomNumber].pos_ != "DET" and x[
                    randomNumber].pos_ != "CCONJ" and x[randomNumber].pos_ != "SCONJ" and \
                        x[randomNumber].pos_ != "PUNCT"):
                    j = i + 1
                    i = i - 1
                    unico = False
            if unico:
                rimozione[i] = randomNumber
        indice = 0
        for i in range(0, int(frase[0][2])):
            NonDaRimuovere = True
            for j in range(0, len(rimozione)):
                if i == rimozione[j]:
                    NonDaRimuovere = False
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and \
                        x[i].pos_ != "PUNCT" and NonDaRimuovere:
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass

    ###secondo pezzo di frase
    if int(frase[1][0]) == 0:
        indice = 0
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 1:
        indice = 0;
        '''
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
        '''
    elif int(frase[1][0]) == 2:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 3:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 4:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 5:
        indice = 0;
        for i in range(int(frase[1][2]), int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and x[
                    i].pos_ != "PUNCT":
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[1][0]) == 6:
        differenza = int(frase[1][1]) - val2
        rimozione = np.zeros(differenza)
        for i in range(0, len(rimozione)):
            randomNumber = np.random.randint(int(frase[0][2]) + 1, int(frase[1][2]))
            unico = True
            for j in range(0, i):
                if rimozione[j] == randomNumber or (
                        x[randomNumber].text != "Drug" and x[randomNumber].pos_ != "DET" and x[
                    randomNumber].pos_ != "CCONJ" and x[randomNumber].pos_ != "SCONJ" and \
                        x[randomNumber].pos_ != "PUNCT"):
                    j = i + 1
                    i = i - 1
                    unico = False
            if unico:
                rimozione[i] = randomNumber
        indice = 0
        for i in range(int(frase[1][2]), int(frase[2][2])):
            NonDaRimuovere = True
            for j in range(0, len(rimozione)):
                if i == rimozione[j]:
                    NonDaRimuovere = False
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and \
                        x[i].pos_ != "PUNCT" and NonDaRimuovere:
                    secondoPezzo[indice, :200] = word_model.wv[x[i].text]
                    secondoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass

    ###terzo pezzo di frase
    if int(frase[2][0]) == 0:
        indice = 0
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 1:
        indice = 0;
        '''
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
        '''
    elif int(frase[2][0]) == 2:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 3:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 4:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 5:
        indice = 0;
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and x[
                    i].pos_ != "PUNCT":
                    terzoPezzo[indice, :200] = word_model.wv[x[i].text]
                    terzoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[2][0]) == 6:
        differenza = int(frase[2][1]) - val3
        rimozione = np.zeros(differenza)
        for i in range(0, len(rimozione)):
            randomNumber = np.random.randint(int(frase[1][2]) + 2, int(frase[2][2]))
            unico = True
            for j in range(0, i):
                if rimozione[j] == randomNumber or (
                        x[randomNumber].text != "Drug" and x[randomNumber].pos_ != "DET" and x[
                    randomNumber].pos_ != "CCONJ" and x[randomNumber].pos_ != "SCONJ" and \
                        x[randomNumber].pos_ != "PUNCT"):
                    j = i + 1
                    i = i - 1
                    unico = False
            if unico:
                rimozione[i] = randomNumber
        indice = 0
        for i in range(int(frase[1][2]) + 1, int(frase[2][2])):
            NonDaRimuovere = True
            for j in range(0, len(rimozione)):
                if i == rimozione[j]:
                    NonDaRimuovere = False
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and \
                        x[i].pos_ != "PUNCT" and NonDaRimuovere:
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    return primoPezzo, secondoPezzo, terzoPezzo

#done
def ritorna_frase_intera_pulita_conDrug(word_model, word_model_4, x, i, frase, val1):
    primoPezzo = np.zeros((val1, 204))
    ###primo pezzo
    if int(frase[0]) == 0:
        indice = 0
        for i in range(0, len(x)):
            try:
                primoPezzo[indice, :200] = word_model.wv[x[i].text]
                primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 2:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 3:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 4:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 5:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and x[
                    i].pos_ != "PUNCT":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 6:
        differenza = int(frase[1]) - val1 #quanti elementi sono in più nella mia frase
        rimozione = np.zeros(differenza)
        for i in range(0, len(rimozione)):
            randomNumber = np.random.randint(0, len(x))
            unico = True
            for j in range(0, i):
                if rimozione[j] == randomNumber or (
                        x[randomNumber].text != "Drug" and x[randomNumber].pos_ != "DET" and x[
                    randomNumber].pos_ != "CCONJ" and x[randomNumber].pos_ != "SCONJ" and \
                        x[randomNumber].pos_ != "PUNCT" and x[randomNumber].text != "PairDrug1" and x[randomNumber].text != "PairDrug2"):
                    j = i + 1
                    i = i - 1
                    unico = False
            if unico:
                rimozione[i] = randomNumber
        indice = 0
        for i in range(0, len(x)):
            NonDaRimuovere = True
            for j in range(0, len(rimozione)):
                if i == rimozione[j]:
                    NonDaRimuovere = False
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and \
                        x[i].pos_ != "PUNCT" and NonDaRimuovere:
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    return primoPezzo

def ritorna_frase_intera_pulita(word_model, word_model_4, x, i, frase, val1):
    primoPezzo = np.zeros((val1, 204))
    ###primo pezzo
    if int(frase[0]) == 0:
        indice = 0
        for i in range(0, len(x)):
            try:
                primoPezzo[indice, :200] = word_model.wv[x[i].text]
                primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 1:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 2:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 3:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 4:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 5:
        indice = 0;
        for i in range(0, len(x)):
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and x[
                    i].pos_ != "PUNCT":
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    elif int(frase[0]) == 6:
        differenza = int(frase[1]) - val1 #quanti elementi sono in più nella mia frase
        rimozione = np.zeros(differenza)
        for i in range(0, len(rimozione)):
            randomNumber = np.random.randint(0, len(x))
            unico = True
            for j in range(0, i):
                if rimozione[j] == randomNumber or (
                        x[randomNumber].text != "Drug" and x[randomNumber].pos_ != "DET" and x[
                    randomNumber].pos_ != "CCONJ" and x[randomNumber].pos_ != "SCONJ" and \
                        x[randomNumber].pos_ != "PUNCT" and x[randomNumber].text != "PairDrug1" and x[randomNumber].text != "PairDrug2"):
                    j = i + 1
                    i = i - 1
                    unico = False
            if unico:
                rimozione[i] = randomNumber
        indice = 0
        for i in range(0, len(x)):
            NonDaRimuovere = True
            for j in range(0, len(rimozione)):
                if i == rimozione[j]:
                    NonDaRimuovere = False
            try:
                if x[i].text != "Drug" and x[i].pos_ != "DET" and x[i].pos_ != "CCONJ" and x[i].pos_ != "SCONJ" and \
                        x[i].pos_ != "PUNCT" and NonDaRimuovere:
                    primoPezzo[indice, :200] = word_model.wv[x[i].text]
                    primoPezzo[indice, 200:] = word_model_4.wv[x[i].text]
                    indice = indice + 1
            except KeyError:
                pass
    return primoPezzo

def divisione_dataset_intelligente_FraseIntera(sents,val):
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    # divisione del dataset in frasi minori di dimensione e maggiori o uguali di dimensione
    lengths = [len(x) for x in sents]  # creo vettore di lunghezze, noi in realtà prima dobbiamo spezzettare
    maxlen = max(lengths)  # prendo il massimo
    X_train1 = np.zeros((len(sents), val, 204))
    X_train = np.zeros((len(sents), 3))
    for i, x in enumerate(sents):
        lunghezza = 0
        if len(x) < val:
            X_train[i][0] = 0
            X_train[i][1] = len(x)
            X_train[i][2] = len(x)
        else:
            lunghezza = 0
            for k in range(0, len(x)):
                if x[k].text != "Drug":
                    lunghezza = lunghezza + 1
            X_train[i][0] = 1
            X_train[i][1] = lunghezza
            if lunghezza > val:
                X_train[i][0] = 2
                lunghezza = 0
                for k in range(0, len(x)):
                    if x[k].text != "Drug" and x[k].pos_ != 'DET':
                        lunghezza = lunghezza + 1
                X_train[i][1] = lunghezza
                if lunghezza > val:
                    X_train[i][0] = 3
                    lunghezza = 0
                    for k in range(0, len(x)):
                        if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ':
                            lunghezza = lunghezza + 1
                    X_train[i][1] = lunghezza
                    if lunghezza > val:
                        X_train[i][0] = 4
                        lunghezza = 0
                        for k in range(0, len(x)):
                            if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[k].pos_ != 'SCONJ':
                                lunghezza = lunghezza + 1
                        X_train[i][0][1] = lunghezza
                        if lunghezza > val:
                            X_train[i][0] = 5
                            lunghezza = 0
                            for k in range(0, len(x)):
                                if x[k].text != "Drug" and x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and \
                                    x[k].pos_ != 'SCONJ' and x[k].pos_ != "PUNCT":
                                    lunghezza = lunghezza + 1
                            X_train[i][1] = lunghezza
                            if lunghezza > val:
                                X_train[i][0] = 6
                                X_train[i][1] = lunghezza
    for i, x in enumerate(sents):
        parte1 = ritorna_frase_intera_pulita(word_model, word_model_4, x, i, X_train[i], val)
        X_train1[i] = parte1
    return X_train1

#done
def divisione_dataset_intelligente_FraseIntera_conDrug(sents,val):
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    # divisione del dataset in frasi minori di dimensione e maggiori o uguali di dimensione
    lengths = [len(x) for x in sents]  # creo vettore di lunghezze, noi in realtà prima dobbiamo spezzettare
    maxlen = max(lengths)  # prendo il massimo
    X_train1 = np.zeros((len(sents), val, 204))
    X_train = np.zeros((len(sents), 3))
    for i, x in enumerate(sents):
        lunghezza = 0
        if len(x) < val:
            X_train[i][0] = 0
            X_train[i][1] = len(x)
            X_train[i][2] = len(x)
        else:
            lunghezza = len(x)
            #for k in range(0, len(x)):
            #    if x[k].text != "Drug":
            #        lunghezza = lunghezza + 1
            #X_train[i][0] = 1
            #X_train[i][1] = lunghezza
            if lunghezza > val:
                X_train[i][0] = 2
                lunghezza = 0
                for k in range(0, len(x)):
                    if x[k].pos_ != 'DET':
                        lunghezza = lunghezza + 1
                X_train[i][1] = lunghezza
                if lunghezza > val:
                    X_train[i][0] = 3
                    lunghezza = 0
                    for k in range(0, len(x)):
                        if x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ':
                            lunghezza = lunghezza + 1
                    X_train[i][1] = lunghezza
                    if lunghezza > val:
                        X_train[i][0] = 4
                        lunghezza = 0
                        for k in range(0, len(x)):
                            if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and x[k].pos_ != 'SCONJ':
                                lunghezza = lunghezza + 1
                        X_train[i][1] = lunghezza
                        if lunghezza > val:
                            X_train[i][0] = 5
                            lunghezza = 0
                            for k in range(0, len(x)):
                                if  x[k].pos_ != 'DET' and x[k].pos_ != 'CCONJ' and \
                                    x[k].pos_ != 'SCONJ' and x[k].pos_ != "PUNCT":
                                    lunghezza = lunghezza + 1
                            X_train[i][1] = lunghezza
                            if lunghezza > val:
                                X_train[i][0] = 6
                                X_train[i][1] = lunghezza
    for i, x in enumerate(sents):
        parte1 = ritorna_frase_intera_pulita_conDrug(word_model, word_model_4, x, i, X_train[i], val)
        X_train1[i] = parte1
    return X_train1



# divisione a 3 di lunghezza variabile in base al massimo
def divisione_dataset1(sents):
    c1max = 0
    c2max = 0
    c3max = 0
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    for i, x in enumerate(sents):
        # divisione della mia frase in 3
        c1 = 0
        c2 = 0
        c3 = 0
        val = 1
        for j in range(0, len(x)):
            if val == 1:
                c1 = c1 + 1
                if x[j].text == "PairDrug1":
                    val = 2
            if val == 2:
                c2 = c2 + 1
                if x[j].text == "PairDrug2":
                    c2 = c2 - 1
                    c3 = c3 + 1
                    val = 3
            if val == 3:
                c3 = c3 + 1
        # TROVO IL MASSIMO DELLA LUNGHEZZA IN PAROLE DI OGNI PEZZO DI FRASE
        if c1 > c1max:
            c1max = c1
        if c2 > c2max:
            c2max = c2
        if c3 > c3max:
            c3max = c3

    X_train1 = np.zeros((len(sents), c1max, 204))
    X_train2 = np.zeros((len(sents), c2max, 204))
    X_train3 = np.zeros((len(sents), c3max, 204))

    for i, sent in enumerate(sents):
        indice1 = 0
        indice2 = 0
        valore = 1
        for j in range(0, len(sent)):
            if valore == 1:
                indice1 = indice1 + 1
                try:
                    X_train1[i, j, :200] = word_model.wv[sent[j].text]
                    X_train1[i, j, 200:] = word_model_4.wv[sent[j].text]
                except KeyError:
                    pass
                if "PairDrug1" in sent[j].text:
                    valore = 2
            if valore == 2:
                if "PairDrug2" in sent[j].text:
                    valore = 3
                    try:
                        X_train3[i, j - indice1 - indice2, :200] = word_model.wv[sent[j].text]
                        X_train3[i, j - indice1 - indice2, 200:] = word_model_4.wv[sent[j].text]
                    except KeyError:
                        pass
                else:
                    indice2 = indice2 + 1
                    try:
                        X_train2[i, j - indice1, :200] = word_model.wv[sent[j].text]
                        X_train2[i, j - indice1, 200:] = word_model_4.wv[sent[j].text]
                    except KeyError:
                        pass
            if valore == 3:
                try:
                    X_train3[i, j - indice1 - indice2, :200] = word_model.wv[sent[j].text]
                    X_train3[i, j - indice1 - indice2, 200:] = word_model_4.wv[sent[j].text]
                except KeyError:
                    pass
    return X_train1, X_train2, X_train3, c1max, c2max, c3max


# divisione a 3 di lunghezza variabile in base al massimo rimuovendo drug e in caso articoli e congiunzioni
def divisione_dataset1conRimozione(sents):
    c1max = 0
    c2max = 0
    c3max = 0
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    for i, x in enumerate(sents):
        # divisione della mia frase in 3
        c1 = 0
        c2 = 0
        c3 = 0
        val = 1
        for j in range(0, len(x)):
            if val == 1:
                if (x[j].pos_ != "det" and x[j].pos_ != "conj" and x[j].text != "Drug"):
                    c1 = c1 + 1
                    if x[j].text == "PairDrug1":
                        val = 2
            if val == 2:
                if (x[j].pos_ != "det" and x[j].pos_ != "conj" and x[j].text != "Drug"):
                    c2 = c2 + 1
                    if x[j].text == "PairDrug2":
                        c2 = c2 - 1
                        c3 = c3 + 1
                        val = 3
            if val == 3:
                if (x[j].pos_ != "det" and x[j].pos_ != "conj" and x[j].text != "Drug"):
                    c3 = c3 + 1
        # TROVO IL MASSIMO DELLA LUNGHEZZA IN PAROLE DI OGNI PEZZO DI FRASE
        if c1 > c1max:
            c1max = c1
        if c2 > c2max:
            c2max = c2
        if c3 > c3max:
            c3max = c3
    print("max 1: " + str(c1max) + " max2: " + str(c2max) + " max3: " + str(c3max))

    X_train1 = np.zeros((len(sents), c1max, 204))
    X_train2 = np.zeros((len(sents), c2max, 204))
    X_train3 = np.zeros((len(sents), c3max, 204))
    
    for i, sent in enumerate(sents):
        indice1 = 0
        indice2 = 0
        valore = 1
        for j in range(0, len(sent)):
            if valore == 1:
                indice1 = indice1 + 1
                try:
                    X_train1[i, j, :200] = word_model.wv[sent[j].text]
                    X_train1[i, j, 200:] = word_model_4.wv[sent[j].text]
                except KeyError:
                    pass
                if "PairDrug1" in sent[j].text:
                    valore = 2
            if valore == 2:
                if "PairDrug2" in sent[j].text:
                    valore = 3
                    try:
                        X_train3[i, j - indice1 - indice2, :200] = word_model.wv[sent[j].text]
                        X_train3[i, j - indice1 - indice2, 200:] = word_model_4.wv[sent[j].text]
                    except KeyError:
                        pass
                else:
                    indice2 = indice2 + 1
                    try:
                        X_train2[i, j - indice1, :200] = word_model.wv[sent[j].text]
                        X_train2[i, j - indice1, 200:] = word_model_4.wv[sent[j].text]
                    except KeyError:
                        pass
            if valore == 3:
                try:
                    X_train3[i, j - indice1 - indice2, :200] = word_model.wv[sent[j].text]
                    X_train3[i, j - indice1 - indice2, 200:] = word_model_4.wv[sent[j].text]
                except KeyError:
                    pass
    return X_train1, X_train2, X_train3, c1max, c2max, c3max



########### divisione del dataset in cui divido in lunghe e corte
def divisione_dataset2(length, sents):
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    # divisione del dataset in frasi minori di dimensione e maggiori o uguali di dimensione
    dimensione = 10
    listaFrasiMinori = list()
    listaFrasiMaggiori = list()
    maxLenCorte = 0
    maxLenLunghe = 0
    for i, x in enumerate(sents):
        if len(x) < dimensione:
            listaFrasiMinori += x
            if len(x) > maxLenCorte:
                maxLenCorte = len(x)
        else:
            listaFrasiMaggiori += x
            if len(x) > maxLenLunghe:
                maxLenLunghe = len(x)
    X_train1 = np.zeros((len(listaFrasiMinori), maxLenCorte, 204))
    X_train2 = np.zeros((len(listaFrasiMaggiori), maxLenLunghe, 204))
    for i, sent in enumerate(listaFrasiMinori):
        for j in range(0, len(sent)):
            X_train1[i, j, :200] = word_model.wv[sent[j].text]
            X_train1[i, j, 200:] = word_model_4.wv[sent[j].text]
    for i, sent in enumerate(listaFrasiMaggiori):
        for j in range(0, len(sent)):
            X_train2[i, j, :200] = word_model.wv[sent[j].text]
            X_train2[i, j, 200:] = word_model_4.wv[sent[j].text]
    return X_train1, X_train2


######### frase unica in cui elimino drug, congiunzioni e articoli e pongo il max lenght a val
def divisione_dataset3(sents,val):
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    # divisione del dataset in frasi minori di dimensione e maggiori o uguali di dimensione
    lengths = [len(x) for x in sents]  # creo vettore di lunghezze, noi in realtà prima dobbiamo spezzettare
    maxlen = max(lengths)  # prendo il massimo
    X_train1 = np.zeros((len(sents), val, 204))
    lMAX = 0
    for i, sent in enumerate(sents):
        length = 0
        for j in range(0, len(sent)):
            if sent[j].text != "Drug":
                length = length + 1
        if length >= val:
            length = 0
            indice = 0
            for j in range(0, len(sent)):
                if sent[j].pos_ != 'DET' and sent[j].pos_ != 'CCONJ' and sent[j].pos_ != 'SCONJ' and sent[
                    j].text != "Drug":  # PUNCT
                    try:
                        X_train1[i, indice, :200] = word_model.wv[sent[j].text]
                        X_train1[i, indice, 200:] = word_model_4.wv[sent[j].text]
                        length = length + 1
                        indice = indice + 1
                    except KeyError:
                        pass
            if length > lMAX:
                lMAX = length
        else:
            indice = 0
            for j in range(0, len(sent)):
                if sent[j].text != "Drug":
                    try:
                        X_train1[i, indice, :200] = word_model.wv[sent[j].text]
                        X_train1[i, indice, 200:] = word_model_4.wv[sent[j].text]
                        indice = indice + 1
                    except KeyError:
                        pass
    return X_train1, lMAX


######### frase unica in cui non pongo limiti length
def divisione_dataset4(sents,max):
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    # divisione del dataset in frasi minori di dimensione e maggiori o uguali di dimensione
    X_train1 = np.zeros((len(sents), max, 204))
    for i, sent in enumerate(sents):
        for j in range(0, len(sent)):
            try:
                X_train1[i, j, :200] = word_model.wv[sent[j].text]
                X_train1[i, j, 200:] = word_model_4.wv[sent[j].text]
            except KeyError:
                pass
    return X_train1

####### dopo aver calcolato la mia frase a 99 andando a rimuovere eventualmente drug ecc vado a dividerla in 3 parti da 33
def divisione_dataset66(sents):
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    X_train1 = np.zeros((len(sents), 66, 204))
    X_train2 = np.zeros((len(sents), 66, 204))
    X_train3 = np.zeros((len(sents), 66, 204))
    X_train = np.zeros((len(sents), 198, 204))
    lengths = [len(x) for x in sents]  # creo vettore di lunghezze, noi in realtà prima dobbiamo spezzettare
    maxlen = max(lengths)  # prendo il massimo
    lMAX = 0
    for i, sent in enumerate(sents):
        length = 0
        for j in range(0, len(sent)):
            length = length + 1
        if length >= 198:
            length = 0
            indice = 0
            for j in range(0, len(sent)):
                if sent[j].pos_ != 'DET' and sent[j].pos_ != 'CCONJ' and sent[j].pos_ != 'SCONJ' and sent[j].pos_ != 'PUNCT':  # PUNCT
                    try:
                        X_train[i, indice, :200] = word_model.wv[sent[j].text]
                        X_train[i, indice, 200:] = word_model_4.wv[sent[j].text]
                        length = length + 1
                        indice = indice + 1
                    except KeyError:
                        pass
            if length > lMAX:
                lMAX = length
        else:
            indice = 0
            for j in range(0, len(sent)):
                try:
                    X_train[i, indice, :200] = word_model.wv[sent[j].text]
                    X_train[i, indice, 200:] = word_model_4.wv[sent[j].text]
                    indice = indice + 1
                except KeyError:
                    pass
  
    for i in range(0, len(sents)):
        indice = 0
        indice2 = 0
        for j in range (0,len(X_train[i])):
            if j<66:
                X_train1[i][j] = X_train[i][j]
            if j>=66 and j<132:
                X_train2[i][indice] = X_train[i][j]
                indice = indice + 1
            if j>=132:
                X_train3[i][indice2] = X_train[i][j]
                indice2 = indice2 + 1
    return X_train1, X_train2, X_train3

def divisione_dataset_due_canali(sents, maxlen):
    word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
    word_model_4 = Word2Vec.load('pos_embedding_four_sized.model')
    # divisione del dataset in frasi minori di dimensione e maggiori o uguali di dimensione
    if maxlen % 2 == 0:
        maxlen1 = int(maxlen/2)
    else:
        maxlen1 = int((maxlen-1)/2) + 1
    X_train = np.zeros((len(sents), maxlen, 204))
    X_train1 = np.zeros((len(sents), maxlen1, 204))
    X_train2 = np.zeros((len(sents), maxlen1, 204))
    for i, sent in enumerate(sents):
        for j in range(0, len(sent)):
            try:
                X_train[i, j, :200] = word_model.wv[sent[j].text]
                X_train[i, j, 200:] = word_model_4.wv[sent[j].text]
            except KeyError:
                pass
    for i in range(0, len(sents)):
        indice = 0
        for j in range(0, maxlen):
            if j < maxlen1:
                X_train1[i][j] = X_train[i][j]
            else:
                X_train2[i][indice] = X_train[i][j]
                indice = indice + 1
    return X_train1, X_train2, maxlen1

def main():
    # caricamento da preprocessing
    sents, labels, v_sents, v_labels, t_sents, t_labels = preprocessing()

    ############## calcolo degli indici statistici
    '''
    # calcolo indici statistici
    #############################################    Inizio calcolo degli indici
    c1max = 0
    c2max = 0
    c3max = 0
    c1min = 10000
    c2min = 10000
    c3min = 10000
    sommaDelleC1 = 0
    sommaDelleC2 = 0
    sommaDelleC3 = 0
    sommaTot=0
    #divido per trovare massimo, minimo, media per i pezzi di frase
    for i, x in enumerate(sents):
        # divisione della mia frase in 3
        c1 = 0
        c2 = 0
        c3 = 0
        val = 1
        for j in range(0, len(x)):
            if x[j].text != "Drug":
                if val == 1:
                    c1 = c1 + 1
                    if x[j].text == "PairDrug1":
                        val = 2
                if val == 2:
                    c2 = c2 + 1
                    if x[j].text == "PairDrug2":
                        c2 = c2 - 1
                        c3 = c3 + 1
                        val = 3
                if val == 3:
                    c3 = c3 + 1
        #TROVO IL MASSIMO DELLA LUNGHEZZA IN PAROLE DI OGNI PEZZO DI FRASE
        if c1 > c1max:
            c1max = c1
        if c2 > c2max:
            c2max = c2
        if c3 > c3max:
            c3max = c3
        # TROVO IL MINIMO DELLA LUNGHEZZA IN PAROLE DI OGNI PEZZO DI FRASE
        if c1 < c1min:
            c1min = c1
        if c2 < c2min:
            c2min = c2
        if c3 < c3min:
            c3min = c3
        #AUMENTO AD OGNI CICLO LA SOMMADELLEC1
        sommaDelleC1 = sommaDelleC1 + c1
        sommaDelleC2 = sommaDelleC2 + c2
        sommaDelleC3 = sommaDelleC3 + c3
        sommaTot = sommaTot + c1 +c2 +c3

    #TROVO LA MEDIA
    numeroDelleFrasi = len(sents)
    avgPrimoPezzo = sommaDelleC1/numeroDelleFrasi
    avgSecondoPezzo = sommaDelleC2/numeroDelleFrasi
    avgTerzoPezzo = sommaDelleC3/numeroDelleFrasi

    #TROVO TUTTO DELLA FRASE COMPLETA:
    lengths = [len(x) for x in sents]  # creo vettore di lunghezze, noi in realtà prima dobbiamo spezzettare
    maxlen = max(lengths)  # prendo il massimo
    minlen = min(lengths)  # prendo il minimo
    avgFrase = sommaTot / numeroDelleFrasi


    v1 = np.zeros(c1max+1)
    v2 = np.zeros(c2max+1)
    v3 = np.zeros(c3max+1)
    v1new = np.zeros(c1max+1)
    v2new = np.zeros(c2max + 1)
    v3new = np.zeros(c3max + 1)

    vfrase = np.zeros(maxlen+1)
    vfrasenew = np.zeros(maxlen + 1)
    #calcolo gli istogrammi dei pezzi di frasi andando a suddividere i miei valori
    for i, x in enumerate(sents):
        # divisione della mia frase in 3
        c1 = 0
        c2 = 0
        c3 = 0
        val = 1
        for j in range(0, len(x)):
            if x[j].text != "Drug":
                if val == 1:
                    c1 = c1 + 1
                    if x[j].text == "PairDrug1":
                        val = 2
                if val == 2:
                    c2 = c2 + 1
                    if x[j].text == "PairDrug2":
                        val = 3
                        c2 = c2 - 1
                        c3 = c3 + 1
                if val == 3:
                    c3 = c3 + 1
        v1[c1] = v1[c1] + 1
        v2[c2] = v2[c2] + 1
        v3[c3] = v3[c3] + 1
    #calcolo freq cumulate
    v1new[0] = v1[0]
    v2new[0] = v2[0]
    v3new[0] = v3[0]
    for z in range(1, len(v1new)):
        v1new[z] = v1[z] + v1new[z-1]
    for k in range(1, len(v2new)):
        v2new[k] = v2[k] + v2new[k - 1]
    for s in range(1, len(v3new)):
        v3new[s] = v3[s] + v3new[s-1]

    #TROVO MEDIANA PRIMO QUARTILE E TERZO QUARTILE SUL VETTORE 1 CUMULATO
    bool1 = True
    bool2 = True
    bool3 = True
    bool4 = True
    bool5 = True
    for i in range(1, len(v1new)):
        if (v1new[i] > numeroDelleFrasi/4) and bool1:
            primoQuartileVettore1 = i-1
            bool1 = False
        if (v1new[i] > numeroDelleFrasi / 2) and bool2:
            medianaVettore1 = i - 1
            bool2 = False
        if (v1new[i] > numeroDelleFrasi*3/4) and bool3:
            terzoQuartileVettore1 = i-1
            bool3 = False
        if (v1new[i] > numeroDelleFrasi*85/100) and bool4:
            quartoPercentileVettore1 = i-1
            bool4 = False
        if (v1new[i] > numeroDelleFrasi*95/100) and bool5:
            quintoPercentileVettore1 = i-1
            bool5 = False

    # TROVO MEDIANA PRIMO QUARTILE E TERZO QUARTILE SUL VETTORE 2 CUMULATO
    bool1 = True
    bool2 = True
    bool3 = True
    bool4 = True
    bool5 = True
    for i in range(1, len(v2new)):
        if (v2new[i] > numeroDelleFrasi / 4) and bool1:
            primoQuartileVettore2 = i - 1
            bool1 = False
        if (v2new[i] > numeroDelleFrasi / 2) and bool2:
            medianaVettore2 = i - 1
            bool2 = False
        if (v2new[i] > numeroDelleFrasi * 3 / 4) and bool3:
            terzoQuartileVettore2 = i - 1
            bool3 = False
        if (v2new[i] > numeroDelleFrasi * 85 / 100) and bool4:
            quartoPercentileVettore2 = i - 1
            bool4 = False
        if (v2new[i] > numeroDelleFrasi * 95 / 100) and bool5:
            quintoPercentileVettore2 = i - 1
            bool5 = False
    # TROVO MEDIANA PRIMO QUARTILE E TERZO QUARTILE SUL VETTORE 3 CUMULATO
    bool1 = True
    bool2 = True
    bool3 = True
    bool4 = True
    bool5 = True
    for i in range(1, len(v3new)):
        if (v3new[i] > numeroDelleFrasi / 4) and bool1:
            primoQuartileVettore3 = i - 1
            bool1 = False
        if (v3new[i] > numeroDelleFrasi / 2) and bool2:
            medianaVettore3 = i - 1
            bool2 = False
        if (v3new[i] > numeroDelleFrasi * 3 / 4) and bool3:
            terzoQuartileVettore3 = i - 1
            bool3 = False
        if (v3new[i] > numeroDelleFrasi*85/100) and bool4:
            quartoPercentileVettore3 = i-1
            bool4 = False
        if (v3new[i] > numeroDelleFrasi*95/100) and bool5:
            quintoPercentileVettore3 = i-1
            bool5 = False
    #ESEGUO LE STESSE OPERAZIONI SULLA FRASE TOTALE
    for i, x in enumerate(sents):
        vfrase[len(x)] = vfrase[len(x)] + 1

    vfrasenew[0] = vfrase[0]
    for p in range(1, len(vfrasenew)):
        vfrasenew[p] = vfrase[p] + vfrasenew[p-1]

    # TROVO MEDIANA PRIMO QUARTILE E TERZO QUARTILE SUL VETTORE 3 CUMULATO
    bool1 = True
    bool2 = True
    bool3 = True
    bool4 = True
    bool5 = True
    for i in range(1, len(vfrasenew)):
        if (vfrasenew[i] > numeroDelleFrasi / 4) and bool1:
            primoQuartileFrase = i - 1
            bool1 = False
        if (vfrasenew[i] > numeroDelleFrasi / 2) and bool2:
            medianaFrase = i - 1
            bool2 = False
        if (vfrasenew[i] > numeroDelleFrasi * 3 / 4) and bool3:
            terzoQuartileFrase = i - 1
            bool3 = False
        if (vfrasenew[i] > numeroDelleFrasi*85/100) and bool4:
            quartoPercentileFrase = i-1
            bool4 = False
        if (vfrasenew[i] > numeroDelleFrasi*95/100) and bool5:
            quintoPercentileFrase = i-1
            bool5 = False
    #############################################    1     ########################################################
    #print('v1new 1 : ' + str(v1new[5]) + ' v1new 2 : ' + str(v1new[11]) +' v1new 3 : ' + str(v1new[20]))
    print('Il numero di frasi totali sono: ' + str(numeroDelleFrasi))
    print('Per il primo pezzo di frase: min = ' + str(c1min)+' max = '+str(c1max)+' avg = '+str(round(avgPrimoPezzo)) + 
    ' primo quartile = '+str(primoQuartileVettore1) + ' mediana = '+str(medianaVettore1) 
    +' terzo quartile = '+ str(terzoQuartileVettore1)
    + '85% percentile = '+ str(quartoPercentileVettore1)
    + '95% percentile = '+ str(quintoPercentileVettore1))

    plt.plot(v1new)
    plt.ylabel('freq cumulate primo pezzo')
    plt.show()
    plt.plot(v1)
    plt.ylabel('Istogramma primo pezzo')
    plt.show()

    #############################################    2     ########################################################
    print('Per il secondo pezzo di frase: min = ' + str(c2min) + ' max = ' + str(c2max) 
    + ' avg = ' + str(round(avgSecondoPezzo)) + ' primo quartile = ' + str(primoQuartileVettore2) 
    + ' mediana = ' + str(medianaVettore2) + ' terzo quartile = ' + str(terzoQuartileVettore2)
    + '85% percentile = ' + str(quartoPercentileVettore2)
    + '95% percentile = ' + str(quintoPercentileVettore2))
    plt.plot(v2new)
    plt.ylabel('freq cumulate secondo pezzo')
    plt.show()
    plt.plot(v2)
    plt.ylabel('Istogramma secondo pezzo')
    plt.show()

    #############################################    3     ########################################################
    print('Per il terzo pezzo di frase: min = ' + str(c3min) + ' max = ' + str(c3max) 
    + ' avg = ' + str(round(avgTerzoPezzo)) + ' primo quartile = ' + str(primoQuartileVettore3) 
    + ' mediana = ' + str(medianaVettore3) + ' terzo quartile = ' + str(terzoQuartileVettore3)
    + '85% percentile = ' + str(quartoPercentileVettore3)
    + '95% percentile = ' + str(quintoPercentileVettore3))
    plt.plot(v3new)
    plt.ylabel('freq cumulate terzo pezzo')
    plt.show()
    plt.plot(v3)
    plt.ylabel('Istogramma terzo pezzo')
    plt.show()

    #############################################    Frase     ########################################################
    #print('vfrase[0]: ' + str(vfrase[20]))
    print('Per la frase intera: min = ' + str(minlen) + ' max = ' + str(maxlen) 
    + ' avg = ' + str(round(avgFrase)) + ' primo quartile = ' + str(primoQuartileFrase) 
    + ' mediana = ' + str(medianaFrase) + ' terzo quartile = ' + str(terzoQuartileFrase)
    + '85% percentile = ' + str(quartoPercentileFrase)
    + '95% percentile = ' + str(quintoPercentileFrase))
    plt.plot(vfrase)
    plt.ylabel('Istogramma frase intera')
    plt.show()
    plt.plot(vfrasenew)
    plt.ylabel('freq cumulate frase intera')
    plt.show()
    #############################################    Fine calcolo degli indici
    '''

    ########################## esperimento rete a 1 canale tagliata a 85
    '''
    lstm_units = 200
    dropout = 0.35
    r_dropout = 0.35
    epochs = 15
    batch_size = 128

    X_train1, max1 = divisione_dataset3(sents,85)
    print(max1)
    X_val1, max2 = divisione_dataset3(v_sents,85)
    print(max2)
    X_test1, max3 = divisione_dataset3(t_sents,85)
    print(max3)

    model1 = network_1_85(lstm_units, dropout, r_dropout)
    history = model1.fit([X_train1], labels,
                         validation_data=([X_val1], v_labels), batch_size=batch_size, epochs=epochs,
                         verbose=1)
    predictions = model1.predict([X_test1])
    numeric_predictions = np.argmax(predictions, axis=1)
    model1.save("modello.network1_85")
    matrix, report, overall_precision, overall_recall, overall_f_score = metrics(t_labels, numeric_predictions)
    plot("Grafici", "Grafico_Rete_1_85", history)
    '''
    ########################## esperimento rete a 1 canale tagliata a 85
    '''
    lstm_units = 150
    dropout = 0.5
    r_dropout = 0.5
    epochs = 50
    batch_size = 128

    X_train1, max1 = divisione_dataset_intelligente_FraseIntera(sents,85)
    print(max1)
    X_val1, max2 = divisione_dataset_intelligente_FraseIntera(v_sents,85)
    print(max2)
    X_test1, max3 = divisione_dataset_intelligente_FraseIntera(t_sents,85)
    print(max3)

    model1 = network_1_85(lstm_units, dropout, r_dropout)
    history = model1.fit([X_train1], labels,
                         validation_data=([X_val1], v_labels), batch_size=batch_size, epochs=epochs,
                         verbose=1)
    predictions = model1.predict([X_test1])
    numeric_predictions = np.argmax(predictions, axis=1)
    model1.save("modelli/network1_85")
    matrix, report, overall_precision, overall_recall, overall_f_score = metrics(t_labels, numeric_predictions)
    plot("Grafici", "Grafico_Rete_1_85", history)
    '''
    ########################## esperimento rete a 1 canale tagliata a 95
    '''
    lstm_units = 200
    dropout = 0.35
    r_dropout = 0.35
    epochs = 15
    batch_size = 128

    X_train1, max1 = divisione_dataset3(sents,95)
    print(max1)
    X_val1, max2 = divisione_dataset3(v_sents,95)
    print(max2)
    X_test1, max3 = divisione_dataset3(t_sents,95)
    print(max3)

    model1 = network_1_95(lstm_units, dropout, r_dropout)
    history = model1.fit([X_train1], labels,
                         validation_data=([X_val1], v_labels), batch_size=batch_size, epochs=epochs,
                         verbose=1)
    predictions = model1.predict([X_test1])
    numeric_predictions = np.argmax(predictions, axis=1)
    model1.save("modello.network1_95")
    matrix, report, overall_precision, overall_recall, overall_f_score = metrics(t_labels, numeric_predictions)
    plot("Grafici", "Grafico_Rete_1_95", history)
    '''
    #####################   esperimento rete divisa in base alla posizione del farmaco                  2
    '''
    lstm_units = 100
    dropout = 0.35
    r_dropout = 0.35
    epochs = 40
    batch_size = 128
    val1 = 80
    val2 = 50
    val3 = 60

    X_train1, X_train2, X_train3 = verifica_Dimensione_divisione_frase_conDrug(sents, val1, val2, val3)
    X_val1, X_val2, X_val3 = verifica_Dimensione_divisione_frase_conDrug(v_sents, val1, val2, val3)
    X_test1, X_test2, X_test3 = verifica_Dimensione_divisione_frase_conDrug(t_sents, val1, val2, val3)

    model1 = network1(val1, val2, val3, lstm_units, dropout, r_dropout)
    history = model1.fit([X_train1, X_train2, X_train3], labels,
                         validation_data=([X_val1, X_val2, X_val3], v_labels), batch_size=batch_size, epochs=epochs,
                         verbose=1)
    model1.save("network_Drug_402530_40epAdaelta")
    predictions = model1.predict([X_test1, X_test2, X_test3])
    numeric_predictions = np.argmax(predictions, axis=1)
    matrix, report, overall_precision, overall_recall, overall_f_score = metrics(t_labels, numeric_predictions)
    plot("Grafici", "Grafico_network_Drug_402530_40epAdaelta", history)
    print(report)
    print(overall_precision)
    print(overall_recall)
    print(overall_f_score)
    print(matrix)
    '''
    ######################### esperimento rete di lunghezza random                  3
    '''
    lstm_units = 100
    dropout = 0.35
    r_dropout = 0.35
    epochs = 40
    batch_size = 128
    val1 = 66
    val2 = 66
    val3 = 66

    X_train1, X_train2, X_train3 = divisione_dataset66(sents)
    X_val1, X_val2, X_val3 = divisione_dataset66(v_sents)
    X_test1, X_test2, X_test3 = divisione_dataset66(t_sents)

    model1 = network1(val1, val2, val3, lstm_units, dropout, r_dropout)
    history = model1.fit([X_train1, X_train2, X_train3], labels,
                         validation_data=([X_val1, X_val2, X_val3], v_labels), batch_size=batch_size, epochs=epochs,
                         verbose=1)
    model1.save("network3_666666_Adaelta")
    predictions = model1.predict([X_test1, X_test2, X_test3])
    numeric_predictions = np.argmax(predictions, axis=1)
    matrix, report, overall_precision, overall_recall, overall_f_score = metrics(t_labels, numeric_predictions)
    plot("Grafici", "Grafico_network_666666_Adaelta", history)
    print(report)
    print(overall_precision)
    print(overall_recall)
    print(overall_f_score)
    print(matrix)
    '''
    ######################### esperimento rete con due canali                3
    lstm_units = 100
    dropout = 0.35
    r_dropout = 0.35
    epochs = 100
    batch_size = 128
    
    lengths = [len(x) for x in sents  + t_sents]  # creo vettore di lunghezze, noi in realtà prima dobbiamo spezzettare
    maxlen = max(lengths)  # prendo il massimo
    X_train1, X_train2, maxlen1 = divisione_dataset_due_canali(sents,maxlen)
    #X_val1, X_val2, maxlen1 = divisione_dataset_due_canali(v_sents,maxlen)
    X_test1, X_test2, maxlen1 = divisione_dataset_due_canali(t_sents,maxlen)
    x = [85,90,95]
    for i in x:
      print("Rete a due canali di lunghezza LSTM pari a: "+str(i))
      model1 = network2Canali(maxlen1, maxlen1, i, dropout, r_dropout)
      history = model1.fit([X_train1, X_train2], labels,
                           validation_data=([X_test1, X_test2], t_labels), batch_size=batch_size, epochs=epochs,
                           verbose=0)
      model1.save("network/network3_2Canali_"+str(i))
      predictions = model1.predict([X_test1, X_test2])
      numeric_predictions = np.argmax(predictions, axis=1)
      matrix, report, overall_precision, overall_recall, overall_f_score = metrics(t_labels, numeric_predictions)
      plot("Grafici", "Grafico_network_2Canali"+str(i), history)
      print(report)
      print(overall_precision)
      print(overall_recall)
      print(overall_f_score)
      print(matrix)
    
    ######################### esperimento rete di lunghezza massima con padding         1
    '''
    lstm_units = 100
    dropout = 0.35
    r_dropout = 0.35
    epochs = 50
    batch_size = 128
    lengths = [len(x) for x in sents + t_sents]  # creo vettore di lunghezze, noi in realtà prima dobbiamo spezzettare
    maxlen = max(lengths)  # prendo il massimo
    X_train1 = divisione_dataset4(sents,maxlen)
    #X_val1 = divisione_dataset4(v_sents,maxlen)
    X_test1 = divisione_dataset4(t_sents,maxlen)
    x = [60,65,70,75,80,85,90,95,100]
    
    for i in x:
      print("Rete a due canali di lunghezza LSTM pari a: "+str(i))
      model1 = network2(maxlen, i, dropout, r_dropout)
      history = model1.fit([X_train1], labels,
                           validation_data=([X_test1], t_labels), batch_size=batch_size, epochs=epochs,
                           verbose=0)
      predictions = model1.predict([X_test1])
      numeric_predictions = np.argmax(predictions, axis=1)
      model1.save("network/network_noMax_50ep_"+str(i))
      matrix, report, overall_precision, overall_recall, overall_f_score = metrics(t_labels, numeric_predictions)
      plot("Grafici", "Grafico_Rete_noMax_50ep"+str(i), history)
      print(report)
      print(overall_precision)
      print(overall_recall)
      print(overall_f_score)
      print(matrix)
    '''
    ######################### esperimento rete di lunghezza 85  con doppio lstm         4
    '''
    lstm_units = 100
    dropout = 0.35
    r_dropout = 0.35
    epochs = 40
    batch_size = 128
    val = 85
    X_train1 = divisione_dataset_intelligente_FraseIntera_conDrug(sents,val)
    X_val1 = divisione_dataset_intelligente_FraseIntera_conDrug(v_sents,val)
    X_test1 = divisione_dataset_intelligente_FraseIntera_conDrug(t_sents,val)
    model1 = network_1_85_doppioLSTM(lstm_units, dropout, r_dropout)
    history = model1.fit([X_train1], labels,
                         validation_data=([X_val1], v_labels), batch_size=batch_size, epochs=epochs,
                         verbose=1)
    predictions = model1.predict([X_test1])
    numeric_predictions = np.argmax(predictions, axis=1)
    model1.save("network/network_85_2hidden_40ep_Adaelta")
    matrix, report, overall_precision, overall_recall, overall_f_score = metrics(t_labels, numeric_predictions)
    plot("Grafici", "Grafico_Rete_85_2hidden_40ep_Adaelta", history)
    print("Inizio stampa dei vari indici: ")
    print(report)
    print(overall_precision)
    print(overall_recall)
    print(overall_f_score)
    print(matrix)
    '''
    
if __name__ == "__main__":
    main()
