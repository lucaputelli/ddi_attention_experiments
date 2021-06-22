import datetime
import pickle
from os import mkdir
from os.path import exists
from pre_processing_lib import *
from dataset_creation_lib import *
from post_processing_lib import *
from attention_network import attention_model, get_weights_model
import numpy
from PIL import ImageFont, ImageDraw, Image
from scipy import stats


def draw(test, test_attention):
    for i in range(len(test)):
        doc_folder = configuration_folder + '/doc' + str(i)
        if not exists(doc_folder):
            mkdir(doc_folder)
        sentence = test.sents[i]
        prediction = from_num_to_class(predictions[i].argmax())
        label = from_num_to_class(t_selected_labels[i].argmax())
        file_name = 'C_' + label + ' P_' + prediction + '.png'
        word_num = len(sentence)
        weights = test_attention[i][:word_num]
        max_v = max(weights)
        min_v = min(weights)
        norm_weights = numpy.array([(w - min_v) / (max_v - min_v) for w in weights])
        # print(norm_weights)
        img = Image.new('RGB', (2000, 100), (255, 255, 255))
        d = ImageDraw.Draw(img)
        x = 0
        y = 20
        index_1 = -1
        index_2 = -1
        for j in range(word_num):
            if sentence[j].text == 'PairDrug1':
                index_1 = j
            if sentence[j].text == 'PairDrug2':
                index_2 = j
            if index_1 != -1 and index_2 != -1:
                break
        after_two = weights[index_2:]
        if len(after_two) == 0:
            post_avg = 0
        else:
            post_avg = numpy.average(after_two)
        for j in range(word_num):
            word = sentence[j].text
            weight = norm_weights[j]
            r = int(255 * weight)
            d.text((x, y), word, fill=(r, 0, 0), font=font)
            text_width, text_height = d.textsize(word, font=font)
            x += text_width + 5
        img.save(doc_folder + '/' + file_name, 'png')


class ClassifiedInstance:

    def __init__(self, sentence, label, prediction, attention):
        self.sentence = sentence
        self.label = label
        self.prediction = prediction
        self.attention = attention

# Pre-processing
if __name__ == "__main__":
    results_file = open('overall.pkl', 'rb')
    results = pickle.load(results_file)
    best = results[24]
    # sents = get_sentences('Train/Sample')
    sents = get_sentences('Dataset/Train/Overall')
    #v_sents = get_sentences('Dataset/Train/Validation')
    test_sents = get_sentences('Dataset/Test/Overall')
    font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 30)
    # v_sents = get_sentences('Train/Dev')
    # test_sents = get_sentences('Dataset/Test/MedLine')

    instances = get_instances(sents)
    instances = [x for x in instances if x is not None]
    instances = negative_filtering(instances)
    instances = [x for x in instances if x.get_dependency_path() is not None and len(x.get_dependency_path()) > 0]

    '''v_instances = get_instances(v_sents)
    v_instances = [x for x in v_instances if x is not None]
    v_instances = negative_filtering(v_instances)
    v_instances = [x for x in v_instances if x.get_dependency_path() is not None and len(x.get_dependency_path()) > 0]'''

    t_instances = get_instances(test_sents)
    t_selected = [x for x in t_instances if x is not None]
    t_negative = [x for x in t_instances if x is None]
    t_selected, t_negative2 = blind_negative_filtering(t_selected)
    t_negative.extend(t_negative2)
    t_negative.extend([x for x in t_selected if x.get_dependency_path() is None or len(x.get_dependency_path()) == 0])
    t_selected = [x for x in t_selected if x.get_dependency_path() is not None and len(x.get_dependency_path()) > 0]

    sents, labels = get_labelled_instances(instances)
    paths = [[x[0] for x in inst.get_dependency_path()] for inst in instances]
    depth = [[x[1] for x in inst.get_depth_first()] for inst in instances]
    breadth = [[x[1] for x in inst.get_breadth_first()] for inst in instances]

    '''v_sents, v_labels = get_labelled_instances(v_instances)
    v_paths = [[x[0] for x in inst.get_dependency_path()] for inst in v_instances]
    v_depth = [[x[1] for x in inst.get_depth_first()] for inst in v_instances]
    v_breadth = [[x[1] for x in inst.get_breadth_first()] for inst in v_instances]'''

    t_selected_sents, t_selected_labels = get_labelled_instances(t_selected)
    t_selected_paths = [[x[0] for x in inst.get_dependency_path()] for inst in t_selected]
    t_depth = [[x[1] for x in inst.get_depth_first()] for inst in t_selected]
    t_breadth = [[x[1] for x in inst.get_breadth_first()] for inst in t_selected]

    t_negative_sents, t_negative_labels = get_labelled_instances(t_negative)
    # t_negative_paths = [[x[0] for x in inst.get_dependency_path()] for inst in t_negative]

    lengths = [len(x) for x in sents+depth+breadth+t_selected_sents+t_depth+t_breadth]
    dim = max(lengths)

    training = Dataset(sents, depth, breadth, labels, dim)
    test = Dataset(t_selected_sents, t_depth, t_breadth, t_selected_labels, dim)

    combinations = [[True, False, False], [False, True, False], [False, False, True]]
    input_configurations = [[False, False], [True, False], [True, True]]
    folder = '2020_10_19'
    if not exists(folder):
        mkdir(folder)
    for sentence, depth_first, breadth_first in combinations:
        conf = ''
        if sentence:
            conf += 'sent'
        if depth_first:
            conf += 'depth'
        if breadth_first:
            conf += 'breadth'
        for pos_tag, offset in input_configurations:
            input_name = 'word'
            if pos_tag:
                input_name += '_tag'
            if offset:
                input_name += '_offset'
            input_folder = folder + '/'+input_name
            if not exists(input_folder):
                mkdir(input_folder)
            p = {
                 'dimension': stats.randint(50, 100),
                 'dropout': stats.randint(10, 50),
                 'recurrent_dropout': stats.randint(10, 50),
                 }
            for i in range(20):
                lstm = p['dimension'].rvs()
                dropout = (p['dropout'].rvs())/100
                recurrent = (p['recurrent_dropout'].rvs())/100
                name = conf + '_' + str(lstm) + '_' + str(np.round(dropout, 2)) + '_' + str(np.round(recurrent, 2))
                configuration_folder = input_folder + '/' + name
                if not exists(configuration_folder):
                    mkdir(configuration_folder)
                model = attention_model(dim, pos_tag, offset, lstm, dropout, recurrent, 0.001)
                training_input = training.get_input(sentence, depth_first, breadth_first, pos_tag, offset)
                labels = training.labels
                test_input = test.get_input(sentence, depth_first, breadth_first, pos_tag, offset)
                t_selected_labels = test.labels
                history = model.fit(training_input, labels,
                                    validation_data=(test_input, t_selected_labels),
                                    # validation_data=(validation_set, Y_val),
                                    batch_size=128, epochs=20, verbose=2)
                plot(configuration_folder, 'loss_accuracy_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), history)
                # training_input = [training_set.input_matrix, training_set.tag_matrix, test_set.label_matrix]
                intermediate_model = get_weights_model(model)
                test_attention = intermediate_model.predict(test_input)
                # Prediction
                predictions = model.predict(test_input)
                averages = 0
                after_two_averages = 0
                numeric_predictions = np.argmax(predictions, axis=1)
                numeric_labels = np.argmax(t_selected_labels, axis=1)
                output = list()
                for i in range(len(test)):
                    att = test_attention[i]
                    pred = numeric_predictions[i]
                    label = numeric_predictions[i]
                    s = test.sents[i]
                    c = ClassifiedInstance(sentence=s, attention=att, prediction=pred, label=label)
                    output.append(c)
                out_file = open(configuration_folder + '/output_file.pkl', 'wb')
                pickle.dump(output, out_file)
                numeric_negative_labels = np.argmax(t_negative_labels, axis=1)
                total_labels = np.concatenate((numeric_labels, numeric_negative_labels))
                total_predictions = np.concatenate((numeric_predictions, np.zeros(len(t_negative_labels), dtype=np.int64)))
                # Metrics
                matrix, report, overall_precision, overall_recall, overall_f_score = metrics(total_labels,
                                                                                             total_predictions)
                f = open(configuration_folder + '/metrics.txt', 'w')
                text = 'Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nOverall precision\n\n{}' \
                       + '\n\nOverall recall\n\n{}\n\nOverall F-score\n\n{}\n'
                f.write(text.format(report, matrix, overall_precision, overall_recall, overall_f_score))
                f.close()
                # Model to JSON
                model_json = model.to_json()
                with open(configuration_folder + '/model.json', "w") as json_file:
                    json_file.write(model_json)

                # Model pickle
                with open(configuration_folder + '/metrics.pickle', 'wb') as pickle_file:
                    pickle.dump([matrix, report, overall_precision, overall_recall, overall_f_score], pickle_file)