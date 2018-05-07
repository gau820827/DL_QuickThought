import re
import random
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

easy_label_map = {0:0, 1:1, 2:2, 3:3, 4:4}

PADDING = "<pad>"
UNKNOWN = "<unk>"
max_seq_length = 20


class Preprocess(object):
    def __init__(self):
        sst_home = '/data/chc631/trees'
        self.training_set = self.load_sst_data(sst_home + '/train.txt')
        self.dev_set = self.load_sst_data(sst_home + '/dev.txt')
        self.test_set = self.load_sst_data(sst_home + '/test.txt')

    def load_sst_data(self,path):
        data = []
        with open(path) as f:
            for i, line in enumerate(f):
                example = {}
                example['label'] = easy_label_map[int(line[1])]
                if example['label'] is None:
                    continue

                # Strip out the parse information and the phrase labels---we don't need those here
                text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
                example['text'] = text[1:]
                data.append(example)

        random.seed(1)
        random.shuffle(data)
        return data


import collections
import numpy as np

def tokenize(string):
    return string.split()

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))

    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices, len(vocabulary)

class Dataset(Preprocess):
    def __init__(self, word_indices):
        Preprocess.__init__(self)
        self.datasets = [self.training_set, self.dev_set, self.test_set]
        self.sentences_to_padded_index_sequences(word_indices, self.datasets)
    def sentences_to_padded_index_sequences(self, word_indices, datasets):
        """
        Annotate datasets with feature vectors. Adding right-sided padding.
        """

        for i, dataset in enumerate(self.datasets):
            for example in dataset:
                example['text_index_sequence'] = torch.zeros(max_seq_length)

                token_sequence = tokenize(example['text'])
                padding = max_seq_length - len(token_sequence)

                for i in range(max_seq_length):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                        pass
                    else:
                        if token_sequence[i] in word_indices:
                            index = word_indices[token_sequence[i]]
                        else:
                            index = word_indices[UNKNOWN]
                    example['text_index_sequence'][i] = index

                example['text_index_sequence'] = example['text_index_sequence'].long().view(1,-1)
                example['label'] = torch.LongTensor([example['label']])


class Dictionary(Preprocess):

    def __init__(self, path=None):
        Preprocess.__init__(self)
        self.word2idx = dict()
        self.idx2word = list()
        # use external dictionary
        if path:
            self.idx2word.append(PADDING)
            self.word2idx[PADDING] = len(self.idx2word) - 1
            self.idx2word.append(UNKNOWN)
            self.word2idx[UNKNOWN] = len(self.idx2word) - 1

            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)
        # build dictionary from training_set
        else:
            word2idx, vocab_size = build_dictionary([self.training_set])
            # sentences_to_padded_index_sequences(word2idx, [training_set, dev_set, test_set])
            idx2word = list(word2idx.keys())
            self.word2idx = word2idx
            self.idx2word = idx2word

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

