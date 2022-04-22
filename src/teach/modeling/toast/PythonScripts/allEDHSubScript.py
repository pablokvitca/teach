import json
import sys
import os
import numpy as np
import unicodedata
import string
import re
import random

from getAllEDHSub import getAllEDHSub

import pickle

import logging
from collections import defaultdict

# from teach.logger import create_logger

SOS_token = 0
EOS_token = 1

# logger = create_logger(__name__, level=logging.INFO)


class Lang:
    def __init__(self, lang_path=None):
        self.SOS_token_index, self.SOS_token = 0, '<SOS>'
        self.EOS_token_index, self.EOS_token = 1, '<EOS>'
        self.PAD_token_index, self.PAD_token = 2, '<PAD>'
        self.UNK_token_index, self.UNK_token = 3, '<UNK>'
        self.word2index = defaultdict(lambda: self.UNK_token_index)
        self.word2count = defaultdict(lambda: 0)
        self.index2word = defaultdict(lambda: self.UNK_token, {
            self.SOS_token_index: self.SOS_token,
            self.EOS_token_index: self.EOS_token,
            self.PAD_token_index: self.PAD_token,
            self.UNK_token_index: self.UNK_token,
        })
        self.n_words = len(self.index2word.keys())  # Count SOS and EOS

        self.loaded_from_file = False
        if lang_path is not None:
            if os.path.exists(lang_path):
                self.loaded_from_file = True
                self.load(lang_path)

    def load(self, lang_path):
        _lang = pickle.load(open(lang_path, 'rb'))
        self.n_words = _lang["n_words"]
        self.SOS_token_index = _lang["SOS_token_index"]
        self.SOS_token = _lang["SOS_token"]
        self.EOS_token_index = _lang["EOS_token_index"]
        self.EOS_token = _lang["EOS_token"]
        self.PAD_token_index = _lang["PAD_token_index"]
        self.PAD_token = _lang["PAD_token"]
        self.UNK_token_index = _lang["UNK_token_index"]
        self.UNK_token = _lang["UNK_token"]
        self.word2index = defaultdict(lambda: self.UNK_token_index, _lang["word2index"])
        self.word2count = defaultdict(lambda: 0, _lang["word2count"])
        self.index2word = defaultdict(lambda: self.UNK_token, _lang["index2word"])

    def save(self, lang_path):
        pickle.dump({
            "word2index": dict(self.word2index),
            "word2count": dict(self.word2count),
            "index2word": dict(self.index2word),
            "n_words": self.n_words,
            "SOS_token_index": self.SOS_token_index,
            "SOS_token": self.SOS_token,
            "EOS_token_index": self.EOS_token_index,
            "EOS_token": self.EOS_token,
            "PAD_token_index": self.PAD_token_index,
            "PAD_token": self.PAD_token,
            "UNK_token_index": self.UNK_token_index,
            "UNK_token": self.UNK_token,
        }, open(lang_path, 'wb'))

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word, override_index=None):
        # if self.loaded_from_file:
            # logger.warning(f"Language is loaded from file, but tried to add word: {word}")
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words if override_index is None else override_index] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

##Splicing
print(sys.argv[1])
path = str(sys.argv[1])


#Get the x,y data
x, y = getAllEDHSub(path)



print("Joining")
x_joined = list(map(lambda l: " ".join(l), x))
y_joined = list(map(lambda l: " ".join(l), y))
# print(x_joined[1:4])

print("Normalizing")
x_normalized = list(map(lambda l: normalizeString(l), x_joined))
y_normalized = list(map(lambda l: normalizeString(l), y_joined))



print(x_normalized[1])

## Pairing ##
def pairUp(x, y):
    pairs = []
    for i in range(len(x)):
        pair = [x[i], y[i]]
        pairs.append(pair)
    return pairs


def readLangs(lang1, lang2, x, y):

    pairs = pairUp(x, y)

    input_lang = Lang()
    output_lang = Lang('teach_output_lang.pickle')

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, x_normalized, y_normalized)
    print("Read %s sentence pairs" % len(pairs))
    
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('log', 'sg')

output_lang.save('teach_output_lang.pickle')
