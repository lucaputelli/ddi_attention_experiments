import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


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
    pyplot.savefig(folder+'/'+name + '.png')


def from_num_to_class(num: int) -> str:
    if num == 0:
        return 'unrelated'
    if num == 1:
        return 'effect'
    if num == 2:
        return 'mechanism'
    if num == 3:
        return 'advise'
    if num == 4:
        return 'int'


# 0 UNRELATED
# 1 EFFECT
# 2 MECHANISM
# 3 ADVISE
# 4 INT
def print_tsv(folder: str, name: str, test_instances, predictions):
    tsv_file = open(folder+'/'+name+'.tsv', 'w')
    for j in range(len(test_instances)):
        pred = predictions[j]
        p_id = test_instances[j].get_pair_id()
        tsv_file.write('\"DDI2013\"\t\"' + p_id + '\"\t\"' + str(pred) + '\"\n')


def print_text_version(folder: str, name: str, test_instances, test_labels, predictions):
    text_file = open (folder+'/'+name + '.txt', 'w')
    for j in range(len(test_instances)):
        actual = np.argmax(test_labels[j])
        predicted = predictions[j]
        doc = test_instances[j].get_doc()
        text_file.write(doc.text + "\t" + from_num_to_class(actual) + "\t" + from_num_to_class(predicted) + "\n")


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


def metrics_2(t_labels, t_predictions):
    numeric_labels = np.argmax(t_labels, axis=1)
    numeric_predictions = np.argmax(t_predictions, axis=1)
    target_names = ['unrelated', 'effect', 'mechanism', 'advise', 'int']
    matrix = confusion_matrix(numeric_labels, numeric_predictions)
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