def does_model_exist(model_load_path):
    pass

def load_model(model_load_path):
    pass

def load_or_create_model(model_load_path, model_class, model_params):
    return load_model(model_load_path) if does_model_exist(model_load_path) else model_class(**model_params)