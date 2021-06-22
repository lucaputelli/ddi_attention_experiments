from pre_processing_lib import get_labelled_instances
from gensim.models import Word2Vec
import numpy as np


word_model = Word2Vec.load('embeddings/pub_med_retrained_ddi_word_embedding_200.model')
tag_model = Word2Vec.load('embeddings/pos_embedding_four_sized.model')


class Dataset:

    def __init__(self, sents, depth, breadth, labels, dim):
        self.sents = sents
        self.labels = labels
        self.depth = depth
        self.breadth = breadth
        X_1 = np.zeros((len(self.sents), dim, 200))
        X_2 = np.zeros((len(self.sents), dim, 200))
        X_3 = np.zeros((len(self.sents), dim, 200))
        pos_1 = np.zeros((len(self.sents), dim, tag_model.vector_size))
        pos_2 = np.zeros((len(self.sents), dim, tag_model.vector_size))
        pos_3 = np.zeros((len(self.sents), dim, tag_model.vector_size))
        d1_1 = np.zeros((len(self.sents), dim, 1))
        d2_1 = np.zeros((len(self.sents), dim, 1))
        d1_2 = np.zeros((len(self.sents), dim, 1))
        d2_2 = np.zeros((len(self.sents), dim, 1))
        d1_3 = np.zeros((len(self.sents), dim, 1))
        d2_3 = np.zeros((len(self.sents), dim, 1))
        for i, sent in enumerate(self.sents):
            index1 = -1
            index2 = -1
            for j in range(len(sent)):
                if sent[j].text == 'PairDrug1':
                    index1 = j
                if sent[j].text == 'PairDrug2':
                    index2 = j
            for j in range(len(sent)):
                try:
                    X_1[i, j, :] = word_model.wv[sent[j].text]
                    pos_1[i, j, :] = tag_model.wv[sent[j].pos_]
                    d1_1[i, j, :] = (j - index1) / len(sent)
                    d2_1[i, j, :] = (j - index2) / len(sent)
                except KeyError:
                    pass
            index1 = -1
            index2 = -2
            for j in range(len(self.depth[i])):
                if self.depth[i][j] == 'pairdrug1':
                    index1 = j
                if self.depth[i][j] == 'pairdrug2':
                    index2 = j
            for j in range(len(self.depth[i])):
                try:
                    X_2[i, j, :] = word_model.wv[self.depth[i][j]]
                    pos = ''
                    for token in sent:
                        if self.depth[i][j] == token.text:
                            pos = token.pos_
                            break
                    pos_2[i, j, :] = tag_model.wv[pos]
                    d1_2[i, j, :] = (j - index1) / len(self.depth[i])
                    d2_2[i, j, :] = (j - index2) / len(self.depth[i])
                except KeyError:
                    pass
            for j in range(len(self.breadth[i])):
                if self.breadth[i][j] == 'pairdrug1':
                    index1 = j
                if self.breadth[i][j] == 'pairdrug2':
                    index2 = j
            for j in range(len(self.breadth[i])):
                try:
                    X_3[i, j, :] = word_model.wv[self.breadth[i][j]]
                    pos = ''
                    for token in sent:
                        if self.breadth[i][j] == token.text:
                            pos = token.pos_
                            break
                    pos_3[i, j, :] = tag_model.wv[pos]
                    d1_3[i, j, :] = (j - index1) / len(self.breadth[i])
                    d2_3[i, j, :] = (j - index2) / len(self.breadth[i])
                except KeyError:
                    pass
        self.sentence_input = [X_1, pos_1, d1_1, d2_1]
        self.depth_input = [X_2, pos_2, d1_2, d2_2]
        self.breadth_input = [X_3, pos_3, d1_3, d2_3]

    def complete_input(self):
        return self.sentence_input + self.depth_input + self.breadth_input

    def get_input(self, sentence: bool = True, depth_first: bool = False, breadth_first: bool = False,
                  pos_tag: bool = True, offset: bool = True):
        input_list = list()
        if pos_tag and offset:
            sentence_features = self.sentence_input
            depth_features = self.depth_input
            breadth_features = self.breadth_input
        elif pos_tag:
            sentence_features = [self.sentence_input[0], self.sentence_input[1]]
            depth_features = [self.depth_input[0], self.depth_input[1]]
            breadth_features = [self.breadth_input[0], self.breadth_input[1]]
        else:
            sentence_features = [self.sentence_input[0]]
            depth_features = [self.depth_input[0]]
            breadth_features = [self.breadth_input[0]]
        if sentence:
            input_list += sentence_features
        if depth_first:
            input_list += depth_features
        if breadth_first:
            input_list += breadth_features
        return input_list

    def __len__(self):
        return len(self.sents)