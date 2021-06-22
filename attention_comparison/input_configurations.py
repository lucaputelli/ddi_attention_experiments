pos_embedding = False
WORD = 'word'
POS_TAG = 'tag'
EDGE = 'edge'
OFFSET = 'offset'
DEPENDENCY_PATH = 'dependency_path'
VISITS = 'visits'
sentence_word = ([WORD], list())
sentence_word_pos = ([WORD, POS_TAG], list())
sentence_complete = ([WORD, POS_TAG, OFFSET], list())
only_word = ([WORD], [WORD])
word_pos = ([WORD, POS_TAG], [WORD, POS_TAG])
word_offset = ([WORD, OFFSET], [WORD])
word_pos_edge = ([WORD, POS_TAG], [WORD, POS_TAG, EDGE])
word_double_pos_edge = ([WORD, POS_TAG], [WORD, POS_TAG, EDGE])
complete = ([WORD, POS_TAG, OFFSET], [WORD, POS_TAG, EDGE])
all_configurations = list()
all_configurations.append(sentence_word)
all_configurations.append(sentence_word_pos)
all_configurations.append(sentence_complete)
all_configurations.append(only_word)
all_configurations.append(word_pos)
all_configurations.append(word_offset)
all_configurations.append(word_pos_edge)
all_configurations.append(word_double_pos_edge)
all_configurations.append(complete)
double_configurations = list()
double_configurations.append(only_word)
double_configurations.append(word_pos)
double_configurations.append(word_pos_edge)
double_configurations.append(word_double_pos_edge)
only_sentence = [[WORD], [WORD, POS_TAG], [WORD, POS_TAG, OFFSET]]