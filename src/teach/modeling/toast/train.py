import logging

from teach.logger import create_logger

logger = create_logger(__name__, level=logging.INFO)

def load_data(data_folder_path, data_splits):
    pass

def init_data_loaders(datasets):
    pass

def main():
    # TODO: load parameters
    model_load_path = None
    data_folder_path = None
    data_splits = ("train", "valid_seen", "valid_unseen")

    # create/load model
    if model_load_path is not None:
        if does_model_exist(model_load_path):
            logger.info(f"Loading model from {model_load_path}.")
        else:
            logger.info(f"Could not find model to load at {model_load_path}. Creating new model.")
    model = load_or_create_model(model_load_path)

    # TODO: load data
    train_dataset, valid_seen_dataset, valid_unseen_dataset = load_data(data_folder_path, data_splits)
    # TODO: encode data - init dataloaders that will encode the data as needed?
    train_dataloader, valid_dataloader, valid_unseen_dataloader = init_data_loaders(
        (train_dataset, valid_seen_dataset, valid_unseen_dataset)
    )

    # TODO: run epoch
    for epoch in trange(epochs):
        train(model, )

    # TODO: run validation
    # TODO: save model if validation better?
    pass


if __name__ == "__main__":
    main()
