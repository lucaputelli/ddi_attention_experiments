from pre_processing_lib import nlp
from spacy import displacy
from pathlib import Path
import networkx as nx
from networkx import NetworkXNoPath, NodeNotFound
import numpy as np


class ClassifiedInstance:

    def __init__(self, text: str, actual: str, prediction: str):
        self.text = text
        self.actual = actual.replace('\n', '')
        self.prediction = prediction.replace('\n', '')
        self.doc = nlp(text)
        # self.dependency_path = self.dependency_path()

    def dependency_path(self):
        html = displacy.render(self.doc, style='dep', page=True)
        output_path = Path('sentence.html')
        output_path.open('w', encoding='utf-8').write(html)
        edges = []
        for token in self.doc:
            # FYI https://spacy.io/docs/api/token
            for child in token.children:
                child_dep = child.dep_
                edges.append(('{0}-{1}'.format(token.lower_, token.i),
                              '{0}-{1}'.format(child.lower_, child.i), child_dep))
        myGraph = nx.Graph()
        for e in edges:
            source = e[0]
            target = e[1]
            label = e[2]
            myGraph.add_edge(source, target, label=label)
        string_drug1 = ''
        string_drug2 = ''
        for i in range(len(self.doc)):
            token = self.doc[i]
            text = token.text
            # Potrei sostituire con in
            if text == 'PairDrug1':
                string_drug1 = text.lower() + '-' + str(i)
            if text == 'PairDrug2':
                string_drug2 = text.lower() + '-' + str(i)
            '''if text != 'PairDrug1' and 'PairDrug1' in text:
                print(text)
            if text != 'PairDrug2' and 'PairDrug2' in text:
                print(text)'''
        try:
            path = nx.shortest_path(myGraph, source=string_drug1, target=string_drug2)
        except NodeNotFound:
            return list()
        except NetworkXNoPath:
            # Non trova il cammino dell'albero sintattico
            return list()
        path_with_labels = list()
        for i in range(len(path) - 1):
            node = path[i]
            node_split = node.rsplit('-')
            next_node = path[i + 1]
            next_split = next_node.rsplit('-')
            edges = myGraph[node]
            for j in edges:
                j_split = j.rsplit('-')
                e = edges[j]
                j_label = e['label']
                if j_label == 'neg':
                    path_with_labels.append((node_split[0], j_split[0], j_label))
            edge = myGraph[node][next_node]
            edge_label = edge['label']
            path_with_labels.append((node_split[0], next_split[0], edge_label))
        # print(path_with_labels)
        return path_with_labels


file = open('word_tag+word_tag_noatt.txt', 'r')
split_error = 0
error_lengths = list()
total_lengths = list()
e_unrelated_lengths = list()
e_ddi_lengths = list()
total_errors = 0
total_distances = list()
error_distances = list()
for line in file:
    split = line.split('\t')
    try:
        instance = ClassifiedInstance(split[0], split[1], split[2])
        length = len(instance.doc)
        total_lengths.append(length)
        d1 = 0
        d2 = 0
        found = False
        for i in range(len(instance.doc)):
            t = instance.doc[i]
            if t.text == 'PairDrug1':
                d1 = i
                found = True
            if t.text == 'PairDrug2':
                d2 = i
        if found:
            d = np.abs(d1-d2)
            total_distances.append(d)
    except IndexError:
        split_error += 1
    if instance.prediction != instance.actual:
        length = len(instance.doc)
        total_errors += 1
        if instance.prediction != 'unrelated':
            e_ddi_lengths.append(length)
        else:
            e_unrelated_lengths.append(length)
        error_lengths.append(length)
        for i in range(len(instance.doc)):
            t = instance.doc[i]
            if t.text == 'PairDrug1':
                d1 = i
                found = True
            if t.text == 'PairDrug2':
                d2 = i
        if found:
            d = np.abs(d1-d2)
            error_distances.append(d)
bins = [0, 15, 30, 45, 60, 75, 90, 105]
hist, edges = np.histogram(total_lengths, bins)
e_hist, e_edges = np.histogram(error_lengths, bins)
ddi_hist, ddi_edges = np.histogram(e_ddi_lengths, bins)
u_hist, u_edges = np.histogram(e_unrelated_lengths, bins)
print(hist)
print(edges)
print(e_hist)
print(e_edges)
percentages = [e_hist[i]/hist[i] for i in range(len(hist))]
print(percentages)
new_bins = [0, 40, 100]
hist, edges = np.histogram(total_lengths, new_bins)
e_hist, e_edges = np.histogram(error_lengths, new_bins)
print(hist)
print(edges)
print(e_hist)
print(e_edges)
print(total_errors)
d_hist, d_edges = np.histogram(total_distances)
ed_hist, ed_edges = np.histogram(error_distances)
print(d_hist)
print(d_edges)
print(ed_hist)
print(ed_edges)
percentages = [ed_hist[i]/d_hist[i] for i in range(len(d_hist))]
print(percentages)