import pickle
from typing import List
import numpy as np
from spacy.tokens import Doc
import networkx as nx
from matplotlib import pyplot as plt


def graph_creation(doc: Doc) -> nx.Graph:
    edges = []
    for token in doc:
        ancestors = list(token.ancestors)
        if len(ancestors) == 0:
            root = token.lower_ + '-' + str(token.i)
        for child in token.children:
            child_dep = child.dep_
            edges.append(('{0}-{1}'.format(token.lower_, token.i),
                          '{0}-{1}'.format(child.lower_, child.i), child_dep))
    my_graph = nx.Graph()
    # ATTENZIONE: se si crea un albero diretto diGraph() NON FUNZIONA IL FILTRO DELLE NEGATIVE
    for e in edges:
        source = e[0]
        target = e[1]
        label = e[2]
        my_graph.add_edge(source, target, label=label)
    return my_graph


class ClassifiedInstance:

    def __init__(self, sentence, label, prediction, attention):
        self.sentence = sentence
        self.label = label
        self.prediction = prediction
        self.attention = attention

    def is_correct(self) -> bool:
        return self.label == self.prediction

    def is_related(self) -> bool:
        return self.prediction != 0

    def __len__(self):
        return len(self.sentence)

    def std_attention(self):
        attention = self.attention[:len(self.sentence)]
        max_val = max(attention)
        min_val = min(attention)
        attention = np.array([(i - min_val)/(max_val - min_val) for i in attention])
        return attention

    def index_of(self, drug_index: int) -> int:
        if drug_index not in [1, 2]:
            raise ValueError
        for i in range(len(self.sentence)):
            if self.sentence[i].text == 'PairDrug' + str(drug_index):
                return i
        return -1

    def get_dependency_path(self):
        graph = graph_creation(self.sentence)
        drug_1 = 'pairdrug1-' + str(self.index_of(1))
        drug_2 = 'pairdrug2-' + str(self.index_of(2))
        try:
            path = nx.shortest_path(graph, source=drug_1, target=drug_2)
        except nx.NodeNotFound:
            return []
        indexes = [int(s.split('-')[-1]) for s in path]
        return indexes


def print_stats(instances):
    filtering_distribution = dict()
    space_distribution = dict()
    mean_list = list()
    before_after = list()
    under_05 = list()
    not_attention = list()
    not_attention_05 = list()
    dependency = list()
    dependency_05 = list()
    path_percentages = list()
    path_covers = list()
    for i in instances:
        att = i.std_attention()
        local_att = [att[j] for j in range(len(att)) if i.index_of(1) <= j <= i.index_of(2)]
        mean_list.append((np.mean(att), np.mean(local_att)))
        filtered_length = len([j for j in att if j > 0.8])
        not_filtered = [j for j in att if j < 0.8]
        not_attention.append(np.mean(not_filtered))
        percentage = filtered_length / len(att)
        path = i.get_dependency_path()
        path_percentage = len([j for j in path if i.index_of(1) <= j <= i.index_of(2)])/len(path)
        path_cover = len([j for j in path if i.index_of(1) <= j <= i.index_of(2)])/(i.index_of(2) - i.index_of(1))
        path_percentages.append(path_percentage)
        path_covers.append(path_cover)
        if len(path) > 0:
            att_dependency = [att[j] for j in range(len(att)) if j in path]
            dependency.append(np.mean(att_dependency))
        space_between_drugs = i.index_of(2) - i.index_of(1)
        if i.index_of(1) > 0 and i.index_of(2) <= len(i) - 1:
            before_att = [att[j] for j in range(len(att)) if j < i.index_of(1)]
            after_att = [att[j] for j in range(len(att)) if j > i.index_of(2)]
            if len(before_att) != 0 and len(after_att) != 0:
                before_after.append((np.mean(before_att),np.mean(after_att)))
        if percentage <= 0.6:
            '''if len(i) >= 40:
                if 'post-marketing' in i.sentence.text:
                    plt.plot(att)
                    # plt.show()
                    # plt.savefig('att_graphics.png')
                    # plt.clf()
                    print(i.sentence)
                    for j in range(len(att)):
                        print(i.sentence[j], att[j])'''
            if i.index_of(1) > 0 and i.index_of(2) <= len(i) - 1:
                local_att = [att[j] for j in range(len(att)) if i.index_of(1) <= j <= i.index_of(2)]
                before_att = [att[j] for j in range(len(att)) if j < i.index_of(1)]
                after_att = [att[j] for j in range(len(att)) if j > i.index_of(2)]
                under_05.append((percentage, np.mean(local_att),np.mean(before_att), np.mean(after_att)))
                not_attention_05.append(np.mean(not_filtered))
                dependency_05.append(np.mean(att_dependency))
        if len(att) not in filtering_distribution.keys():
            filtering_distribution[len(att)] = [percentage]
        else:
            filtering_distribution[len(att)] = filtering_distribution[len(att)] + [percentage]
        if space_between_drugs not in space_distribution.keys():
            space_distribution[space_between_drugs] = [percentage]
            '''if space_between_drugs < 20 and i.is_correct() and i.is_related():
                print(i.sentence)
                plt.plot(att)
                plt.show()
                # plt.savefig('att_graphics.png')
                plt.clf()
                for j in range(len(att)):
                    print(i.sentence[j], att[j])'''
        else:
            space_distribution[space_between_drugs] = space_distribution[space_between_drugs] + [percentage]
    for k in filtering_distribution:
        filtering_distribution[k] = np.mean(filtering_distribution[k])
    under_20 = list()
    under_40 = list()
    under_60 = list()
    over_60 = list()
    for k, v in filtering_distribution.items():
        if k <= 20:
            under_20.append(v)
        elif k <= 40:
            under_40.append(v)
        elif k <= 60:
            under_60.append(v)
        else:
            over_60.append(v)
    print('UNDER 20', np.mean(under_20), 'UNDER 40', np.mean(under_40),
          'UNDER 60', np.mean(under_60), 'OVER 60', np.mean(over_60))
    filter_d = {'<20': np.mean(under_20), '20-40': np.mean(under_40), '40-60': np.mean(under_60), '>60': np.mean(over_60)}
    mean_general = np.mean([k for k, v in mean_list])
    mean_local = np.mean([v for k, v in mean_list])
    print('TOTAL', mean_general, 'LOCAL', mean_local)
    mean_before = np.mean([k for k, v in before_after])
    mean_after = np.mean([v for k, v in before_after])
    print('BEFORE', mean_before, 'AFTER', mean_after)
    stats_05 = [np.mean([j[i] for j in under_05]) for i in range(4)]
    print('Not attention', np.mean(not_attention))
    print('Dependency Path', np.mean(dependency))
    print('UNDER 0.5')
    print('Percentage:', stats_05[0], 'Local:', stats_05[1], 'Before', stats_05[2], 'After', stats_05[3])
    print('Not attention most filtered', np.mean(not_attention_05))
    print('Dependency Path most filtered', np.mean(dependency_05))
    space_20 = list()
    space_30 = list()
    space_45 = list()
    space_over = list()
    for k in space_distribution:
        space_distribution[k] = np.mean(space_distribution[k])
    for k, v in space_distribution.items():
        if k <= 10:
            space_20.append(v)
        elif k <= 20:
            space_30.append(v)
        elif k <= 35:
            space_45.append(v)
        else:
            space_over.append(v)
    space_d = {'<10': np.mean(space_20), '10-20': np.mean(space_30), '20-30': np.mean(space_over), '>30': np.mean(space_over)}
    print('Path percentage:', np.mean(path_percentages))
    print('Path cover:', np.mean(path_covers))
    return filter_d, space_d


f = open('output_file.pkl', 'rb')
instances: List[ClassifiedInstance] = pickle.load(f)
related = [i for i in instances if i.is_related()]
unrelated = [i for i in instances if not i.is_related()]
print('RELATED')
filter_rel, space_rel = print_stats(related)
plt.bar(filter_rel.keys(), filter_rel.values())
plt.ylim(bottom=0, top=1)
plt.savefig('rel_filtering.png')
plt.clf()
plt.bar(space_rel.keys(), space_rel.values())
plt.ylim(bottom=0, top=1)
plt.savefig('rel_space.png')
plt.clf()
print('\nUNRELATED')
filter_un, space_un = print_stats(unrelated)
plt.bar(filter_un.keys(), filter_un.values())
plt.ylim(bottom=0, top=1)
plt.savefig('unrel_filtering.png')
plt.clf()
plt.bar(space_un.keys(), space_un.values())
plt.ylim(bottom=0, top=1)
plt.savefig('unrel_space.png')
