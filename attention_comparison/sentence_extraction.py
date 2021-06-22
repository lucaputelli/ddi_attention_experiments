from os import listdir
from xml.dom import minidom
import pickle
from pre_processing_lib import nlp, substitution, doc_cleaning
from typing import List
from spacy.tokens import Doc


class Entity:

    def __init__(self, id: str, text: str, char_offset: str):
        self.id = id
        self.entity_text = text
        self.char_offset = char_offset


class Pair:

    def __init__(self, e1: str, e2: str, ddi: str, type: str):
        self.e1 = e1
        self.e2 = e2
        self.ddi = ddi
        self.type = type


class Sentence:

    def __init__(self, id: str, text: str, entities, pairs):
        self.id = id
        self.text = text
        self.entities = entities
        self.pairs = pairs


class Instance:

    def __init__(self, sentence: Sentence, pair: Pair, doc: Doc):
        self._sentence = sentence
        self._pair = pair
        self._doc = doc
        self._dependency_path = None
        self._depth_first = None
        self._breadth_first = None

    def get_text(self) -> str:
        return self._sentence.attributes['text'].value

    def get_sentence(self) -> Sentence:
        return self._sentence

    def get_doc(self) -> Doc:
        return self._doc

    def get_sentence_id(self) -> str:
        return self._sentence.id

    def get_class(self) -> str:
        ddi = self._pair.ddi
        if ddi == 'true':
            return self._pair.type
        else:
            return ddi

    def get_dependency_path(self):
        return self._dependency_path

    def set_dependency_path(self, dependency_path):
        self._dependency_path = dependency_path


def get_instances(sentences: List[Sentence]):
    instances = list()
    for i in range(len(sentences)):
        entities: List[Entity] = sentences[i].entities
        entity_tuples = list()
        text = sentences[i].text
        nlp_doc = nlp(text)
        tokens = list(nlp_doc)
        # displacy.serve(nlp_doc, style='dep')
        for j in range(len(entities)):
            e_text = entities[j].entity_text
            offset_string = entities[j].char_offset
            split = str.split(offset_string, ';')
            if len(split) > 1:
                continue
            for s in split:
                offsets = str.split(s, "-")
                left = int(offsets.__getitem__(0))
                right = int(offsets.__getitem__(1))
                entity = (e_text, left, right)
                entity_tuples += [entity]
        # print(entity_tuples)
        for entity in entity_tuples:
            left_tuple = entity[1]
            right_tuple = entity[2]
            for k in range(len(tokens)):
                t = tokens.__getitem__(k)
                left_idx = t.idx
                length = len(t.text)
                right_idx = t.idx + length - 1
                if left_tuple == left_idx:
                    if right_idx == right_tuple:
                        a = 0
                        # print(t)
                    else:
                        n = 1
                        # print(right_tuple, right_idx)
                        while right_idx < right_tuple:
                            if k + n >= len(tokens):
                                break
                            next = tokens.__getitem__(k + n)
                            right_idx = next.idx + len(next.text)
                            n = n + 1
                        if (right_idx - 1) >= right_tuple:
                            span = nlp_doc[k: k + n]
                            span.merge()
                            # print(tokens[k:k + n])
        tokens = nlp_doc
        ents: List[Entity] = sentences[i].entities
        pairs = sentences[i].pairs
        for pair_index in range(len(pairs)):
            new_doc = tokens
            pair = pairs[pair_index]
            e1 = pair.e1
            e2 = pair.e2
            entity_triples = list()
            for l in range(len(ents)):
                etext = entities[l].entity_text
                offset = entities[l].char_offset
                left = int(offset.split("-")[0])
                index = -1
                for m in range(len(tokens)):
                    token = tokens[m]
                    l_idx = token.idx
                    if etext in token.text and left == l_idx:
                        index = m
                        break
                e_id = ents[l].id
                entity_triples += [(e_id, etext, index)]
            # print(entity_triples)
            for (ent_id, text, sub_index) in entity_triples:
                if ent_id == e1:
                    text1 = text
                    sub_index1 = sub_index
                    my_doc = substitution(new_doc, sub_index, 1)
                    #print(my_doc)
                    break
            for (ent_id, text, sub_index) in entity_triples:
                if ent_id == e2:
                    text2 = text
                    sub_index2 = sub_index
                    my_doc = substitution(my_doc, sub_index, 2)
                    #print(my_doc)
                    break
            for (ent_id, text, sub_index) in entity_triples:
                if ent_id != e1 and ent_id != e2:
                    my_doc = substitution(my_doc, sub_index, 0)
                    #print(my_doc)
            if text1.lower() == text2.lower():
                my_doc = substitution(my_doc, sub_index1, -1)
                my_doc = substitution(my_doc, sub_index2, -1)
            my_doc = doc_cleaning(my_doc)
            print(my_doc)
            instance = Instance(sentences[i], pair, my_doc)
            if instance not in instances:
                instances += [instance]
    return instances


def create_sentences_dump(path: str):
    pickle_file = open('sentence.pkl', 'wb')
    files = listdir(path)
    tot_sentences = []
    for f in files:
        doc = minidom.parse(path + "/" + f.title().lower())
        sentences = doc.getElementsByTagName('sentence')
        tot_sentences += sentences
    sentence_list = list()
    for sentence in tot_sentences:
        sentence_id = sentence.attributes['id'].value
        text = sentence.attributes['text'].value
        entities = sentence.getElementsByTagName('entity')
        entity_list = list()
        for j in range(len(entities)):
            id = entities[j].attributes['id'].value
            e_text = entities[j].attributes['text'].value
            offset_string = entities[j].attributes['charOffset'].value
            e = Entity(id, e_text, offset_string)
            entity_list.append(e)
        pairs = sentence.getElementsByTagName('pair')
        pair_list = list()
        for pair in pairs:
            e1 = pair.attributes['e1'].value
            e2 = pair.attributes['e2'].value
            ddi = pair.attributes['ddi'].value
            if ddi == 'false':
                type = 'unrelated'
            else:
                try:
                    type = pair.attributes['type'].value
                except KeyError:
                    type = 'unrelated'
            p = Pair(e1, e2, ddi, type)
            pair_list.append(p)
        s = Sentence(sentence_id, text, entity_list, pair_list)
        sentence_list.append(s)
    pickle.dump(sentence_list, pickle_file)

path = 'Dataset/Train/Overall'
create_sentences_dump(path)