import logging
import os
import pickle
from collections import defaultdict

from teach.logger import create_logger

logger = create_logger(__name__, level=logging.INFO)


class Lang:
    def __init__(self, lang_path=None):
        self.SOS_token_index, self.SOS_token = 0, '<SOS>'
        self.EOS_token_index, self.EOS_token = 1, '<EOS>'
        self.PAD_token_index, self.PAD_token = 2, '<PAD>'
        self.UNK_token_index, self.UNK_token = 3, '<UNK>'
        self.word2index = defaultdict(self._word2index_default)
        self.word2count = defaultdict(self._word2count_default)
        self.index2word = defaultdict(self._index2word_default, {
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

    def _word2index_default(self):
        return self.UNK_token_index

    def _word2count_default(self):
        return 0

    def _index2word_default(self):
        return self.UNK_token

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
        self.word2index = defaultdict(self._word2index_default, _lang["word2index"])
        self.word2count = defaultdict(self._word2count_default, _lang["word2count"])
        self.index2word = defaultdict(self._index2word_default, _lang["index2word"])

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
        if self.loaded_from_file:
            logger.warning(f"Language is loaded from file, but tried to add word: {word}")
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words if override_index is None else override_index] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
