from typing import List
import numpy as np
from itertools import groupby
from operator import itemgetter
from spacy.language import Doc


def online_avg(array: np.ndarray, window: int) -> List[float]:
    """ Given an array it computes its online average of the given window size.

    :param array: The array which online average is to be computed.
    :param window: Window size to compute the online average.
    :return: An array with the online average of the given array.
    """

    array_avg = []
    for i in range(len(array) - window + 1):
        array_avg.append(np.average(array[i:i+window-1]))
    return array_avg


def get_sentences_from_indexes(indexes: List[int], doc: Doc):
    sents = [s for s in doc.sents]
    sentence_set = set()
    for index in indexes:
        token = doc[index]
        for s in sents:
            if token in s:
                sentence_set.add(s)
    return sentence_set


def join_windows(indexes: List[int], window: int) -> List[List[int]]:
    """ Given the indexes of the words and the window size, returns the list of joint documents part
    if any overlapping is occurring.

    :param indexes: List of the indexes of the first word of the document part.
    :param window: Window size to consider
    :return: A list of lists containing the incexes of the words in the document to retrieve.
    """

    new_indexes = []
    # Create all the windows from the original indexes
    for index in indexes:
        new_indexes.append([i for i in range(index, index + window + 1)])
    # Eliminate the duplicate indexes
    index_set = set(indexes)
    for index in new_indexes:
        index_set = index_set.union(set(index))
    # Sort the indexes
    index_set = np.sort(list(index_set))
    # Separate the continuous groups of indexes
    sentences = []
    for key, g in groupby(enumerate(index_set), lambda x: x[0] - x[1]):
        sentences.append(list(map(itemgetter(1), g)))
    return sentences


def join_full_sentences(doc: List[str], doc_part_list: List[List[str]]) -> List[List[int]]:
    """ Join parts of the document and expand so that they form full sentences.

    :param doc: Tokenized document
    :param doc_part_list: Tokenized part of the document
    :return: List of list of indices of the tokens in the document corresponding to sentences.
    """

    sentences = []
    for doc_part in doc_part_list:
        length = len(doc_part)
        start = end = -1
        for index in [i for i, e in enumerate(doc) if e == doc_part[0]]:
            if doc[index:index + length] == doc_part:
                start = index
                end = index + length - 1
                break
        while start > 0 and doc[start] != ".":
            start -= 1
        start += 1
        while end < length - 1 and doc[end] != ".":
            end += 1
        sentences.append([x for x in range(start, end + 1)])
    flattened_list = np.sort(list(set([item for sublist in sentences for item in sublist])))
    sentences = []
    for key, g in groupby(enumerate(flattened_list), lambda x: x[0] - x[1]):
        sentences.append(list(map(itemgetter(1), g)))
    return sentences