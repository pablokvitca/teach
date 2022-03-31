def get_text_tokens_from_instance(edh_instance):
    tokens_list = []
    cleaned_dialog = edh_instance["dialog_history_cleaned"]
    for dialog_part in cleaned_dialog:
        tokens_list.extend(dialog_part[1].split())
    return tokens_list


def encode_as_word_vectors(text_from_instance):
    return []  # TODO: add word2vec code


def pad_list(_list, pad_size, pad_token='<PAD>'):
    return _list + [pad_token] * (pad_size - len(_list))
