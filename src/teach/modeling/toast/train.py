import logging
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from teach.inference.actions import all_agent_actions
from teach.logger import create_logger
from teach.modeling.toast.NaiveDataModule import NaiveDataModule
from teach.modeling.toast.NaiveMultimodalModel import NaiveMultiModalModel

logger = create_logger(__name__, level=logging.INFO)


def does_model_exist(model_load_path):
    return False  # TODO: implement


def load_or_create_model(model_load_path):
    # TODO: implement loading
    if model_load_path is not None:
        if does_model_exist(model_load_path):
            logger.info(f"Loading model from {model_load_path}.")
        else:
            logger.info(f"Could not find model to load at {model_load_path}. Creating new model.")
    return NaiveMultiModalModel(
        [
            {"in_channels": 3, "out_channels": 32, "kernel_size": 11, "stride": 3},
            {"in_channels": 32, "out_channels": 64, "kernel_size": 7},
            {"in_channels": 64, "out_channels": 8, "kernel_size": 5}
        ],  # image_conv_kwargs
        [(30752, 512), (512, 128), (128, 16)],  # image_hidden_layer_sizes
        300,  # text_word_vec_size
        100,  # text_input_words
        [128, 16],  # text_hidden_layer_sizes
        100 * len(all_agent_actions),  # prev_actions_input_size
        [128, 32],  # prev_actions_hidden_layer_sizes
        [64, 32],  # combination_hidden_layers_size
        len(all_agent_actions)  # output_layer_size
    )


def main(data_folder_path, wv2_path, model_checkpoints_path):
    model_load_path = None
    # data_folder_path = '/Volumes/Extreme SSD/teach-dataset/'
    data_folder_path = "/home/sethsclass/teach-dataset/"

    logger.info(f"loading from path: {data_folder_path}")

    naive_datamodule = NaiveDataModule(
        data_folder_path,
        32,  # batch_size,
        x_text_pad_length=100,
        x_prev_action_pad_length=100,
        use_small_dataset=True
    )
    naive_datamodule.setup("fit")
    naive_datamodule.setup("validate")
    logger.info("train and valid have been setup")

    # create/load model
    model = load_or_create_model(model_load_path)
    logger.info("model loaded")

    checkpoint_callback = ModelCheckpoint(dirpath="my/path/", save_top_k=2, monitor="val_loss")
    trainer = Trainer(
        accelerator="cpu",
        # accelerator="gpu", gpus=[0],
        auto_lr_find=True,
        track_grad_norm=2
    )
    logger.info("trainer created")

    logger.info("Tuning training hyperparameters")
    trainer.tune(model, datamodule=naive_datamodule)
    logger.info(f"Trainer tuned. LR: {model.learning_rate}")

    logger.info("Fitting model...")
    trainer.fit(
        model=model,
        datamodule=naive_datamodule
    )

    logger.info("Done!")


if __name__ == "__main__":
    assert len(sys.argv) == 3
    data_folder_path, wv2_path, model_checkpoints_path = sys.argv
    main(data_folder_path, wv2_path, model_checkpoints_path)
