import numpy as np
import torch

from teach.logger import create_logger
logger = create_logger(__name__)


def get_text_tokens_from_instance(edh_instance):
    tokens_list = []
    cleaned_dialog = edh_instance["dialog_history_cleaned"]
    for dialog_part in cleaned_dialog:
        tokens_list.extend(dialog_part[1].split())
    return tokens_list


def encode_as_word_vectors(w2v, text_from_instance, pad_token='<PAD>'):
    return torch.from_numpy(np.stack([
        w2v[word].copy()
        if word != pad_token and word in w2v
        else torch.zeros(w2v.vector_size)
        for word in text_from_instance
    ]))


def pad_list(_list, pad_size, pad_token='<PAD>', max_tokens=None):
    max_tokens = pad_size if max_tokens is None or max_tokens < pad_size else max_tokens
    _list = _list[:max_tokens]
    return _list + [pad_token] * (pad_size - len(_list))
