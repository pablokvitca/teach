import logging
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from teach.inference.actions import all_agent_actions
from teach.logger import create_logger
from teach.modeling.toast.NaiveDataModule import NaiveDataModule
from teach.modeling.toast.NaiveMultimodalModel import NaiveMultiModalModel

logger = create_logger(__name__, level=logging.INFO)


def does_model_exist(model_load_path):
    return os.path.exists(model_load_path)


def load_or_create_model(model_load_path, model_load_name):
    path = os.path.join(model_load_path or '', model_load_name)
    if model_load_path is not None and does_model_exist(path):
        logger.info(f"Loading model from {path}.")
        return NaiveMultiModalModel.load_from_checkpoint(path)
    else:
        logger.info(f"Could not find model to load at {path}. Creating new model.")
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


def main(data_folder_path, wv2_path, model_checkpoints_path, model_load_name):
    logger.info(f"loading from path: {data_folder_path}")
    logger.info(f"Using gensim embeddings from {wv2_path}")
    logger.info(f"Saving/loading model checkpoints to/from {model_checkpoints_path} (model_load_name: {model_load_name})")

    naive_datamodule = NaiveDataModule(
        data_folder_path,
        wv2_path,
        512,  # batch_size,
        x_text_pad_length=100,
        x_prev_action_pad_length=100,
        use_small_dataset=True
    )
    naive_datamodule.setup("fit")
    naive_datamodule.setup("validate")
    logger.info("train and valid have been setup")

    # create/load model
    model = load_or_create_model(model_checkpoints_path, model_load_name)
    logger.info("model loaded")

    checkpoint_callback = ModelCheckpoint(dirpath=model_checkpoints_path, save_top_k=3, monitor="val_loss")
    trainer = Trainer(
        # accelerator="cpu",
        accelerator="gpu", gpus=[0],
        auto_lr_find=True,
        # auto_scale_batch_size=True,
        track_grad_norm=2,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback],
        max_epochs=100,
        num_sanity_val_steps=3,
        detect_anomaly=True,
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

    logger.info(f"Done! Best model at {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    assert len(sys.argv) == 5
    data_folder_path, wv2_path, model_checkpoints_path, model_load_name = sys.argv[1:]
    main(data_folder_path, wv2_path, model_checkpoints_path, model_load_name)
