from vector_lib import *
from scipy import stats
from create_lstm_model import DDIModel
from input_configurations import *

sents = get_sentences('Train/Sample')
v_sents = get_sentences('Train/Sample')
test_sents = get_sentences('Train/Dev')
# sents = get_sentences('Dataset/Train/Onlytrain')
# v_sents = get_sentences('Dataset/Train/Validation')
# test_sents = get_sentences('Dataset/Test/Overall')
instances = get_instances(sents)
instances = negative_filtering(instances)
v_instances = get_instances(v_sents)
v_instances = negative_filtering(v_instances)
t_instances = get_instances(test_sents)
t_selected, t_negative = blind_negative_filtering(t_instances)
sents, labels = get_labelled_instances(instances)
v_sents, v_labels = get_labelled_instances(v_instances)
t_selected_sents, t_selected_labels = get_labelled_instances(t_selected)
t_negative_sents, t_negative_labels = get_labelled_instances(t_negative)
# t_sents, t_labels = get_labelled_instances(t_instances)
word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
tag_model = Word2Vec.load('ddi_pos_embedding.model')
edge_model = Word2Vec.load('ddi_edge_embedding.model')
training_vectors = vectors_composition(instances, word_model, tag_model, edge_model)
validation_vectors = vectors_composition(v_instances, word_model, tag_model, edge_model)
test_vectors = vectors_composition(t_selected, word_model, tag_model, edge_model)
sentence_max_len, path_max_len, vector_size = max_len_calculation(training_vectors, validation_vectors, test_vectors, VISITS)
# Non serve perchè uso positional embedding
if not pos_embedding:
    off_dim = offset_dimension(training_vectors, validation_vectors, test_vectors)
else:
    off_dim = 0
print(sentence_max_len, path_max_len)
training_set = InputSets(training_vectors, word_model.vector_size, tag_model.vector_size,
                           edge_model.vector_size, sentence_max_len, path_max_len)
matrix = training_set.sentence_word_vectors()
print(matrix.shape)
validation_set = InputSets(validation_vectors, word_model.vector_size, tag_model.vector_size,
                           edge_model.vector_size, sentence_max_len, path_max_len)
print(validation_set.sentence_tag_vectors().shape)
test_set = InputSets(test_vectors, word_model.vector_size, tag_model.vector_size,
                           edge_model.vector_size, sentence_max_len, path_max_len)
print(test_set.path_edge_vectors().shape)
p = {'sentence_layers': stats.randint(1, 2),
     'dimension': stats.randint(1, 4),
     'path_dimension': stats.randint(1, 4),
     'dropout': stats.randint(0, 100),
     'recurrent_dropout': stats.randint(0, 100),
     'learning_rate': stats.randint(1, 5),
     'epochs': stats.randint(1, 6)
     }
ddi_model = DDIModel('prova', sentence_max_len=sentence_max_len, path_max_len=path_max_len, off_dim=off_dim,
                     word_model=word_model, tag_model=tag_model, edge_model=edge_model, hyperparameters_dist=p,
                     )
y = {'main_output': labels}
val_y = v_labels
#ddi_model.random_search(training_set, labels, validation_set, val_y, 3)
total_instances = t_negative + t_selected
total_classes = np.concatenate((t_negative_labels, t_selected_labels))
negative_classes = list()
for i_neg in t_negative:
    negative_classes.append(0)
negative_classes = np.array(negative_classes)
ddi_model.random_attention(training_set, labels, validation_set, val_y, test_set, negative_classes, total_instances,
                           total_classes, [complete], 3)
