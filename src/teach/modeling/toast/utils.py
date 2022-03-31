from numpy import zeros


def get_text_tokens_from_instance(edh_instance):
    tokens_list = []
    cleaned_dialog = edh_instance["dialog_history_cleaned"]
    for dialog_part in cleaned_dialog:
        tokens_list.extend(dialog_part[1].split())
    return tokens_list


def encode_as_word_vectors(model, text_from_instance, pad_token='<PAD>'):
    return [
        model.wv[word] if word in model.wv or word != pad_token else zeros(model.vector_size)
        for word in text_from_instance
    ]


def pad_list(_list, pad_size, pad_token='<PAD>'):
    return _list + [pad_token] * (pad_size - len(_list))
