import pickle
from os import listdir
import numpy as np

path = '2019_09_23_duelivelli_adam_15epoche'
files = listdir(path)
att_config = ['context_att', 'att']
input_config = ['word_pos_offset']
for input in input_config:
    attention_dict = dict()
    for att in att_config:
        precision_list = list()
        recall_list = list()
        f_score_list = list()
        for f in files:
            file_path = path+'/'+f+'/'+att+'/'+input+'/metrics.pickle'
            file = open(file_path, 'rb')
            metrics = pickle.load(file)
            precision = metrics[2]
            recall = metrics[3]
            f_score = metrics[4]
            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)
        # attention_dict.__setitem__(att+'_recall', np.max(recall_list))
        attention_dict.__setitem__(att+'_fscore', max(f_score_list))
    print(attention_dict)