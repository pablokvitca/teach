import logging
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from teach.inference.actions import all_agent_actions
from teach.logger import create_logger
from teach.modeling.toast.GRUTextOnlyModel import GRUTextOnlyModel
from teach.modeling.toast.NaiveDataModule import NaiveDataModule
from teach.modeling.toast.NaiveMultimodalModel import NaiveMultiModalModel
from teach.modeling.toast.SequentialDataModule import SequentialDataModule

logger = create_logger(__name__, level=logging.INFO)


def does_model_exist(model_load_path):
    return os.path.exists(model_load_path)


def load_or_create_model(cfg: DictConfig, datamodule):
    path = os.path.join(cfg.model_checkpoints_path or '', cfg.model_load_name)
    if cfg.model_checkpoints_path is not None and does_model_exist(path):
        logger.info(f"Loading model from {path}.")
        if cfg.model_type == 'naive':
            return NaiveMultiModalModel.load_from_checkpoint(path)
        if cfg.model_type == 'gru_text':
            return GRUTextOnlyModel.load_from_checkpoint(path)
    else:
        logger.info(f"Could not find model to load at {path}. Creating new model.")
        if cfg.model_type == 'naive':
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
        if cfg.model_type == 'gru_text':
            return GRUTextOnlyModel(
                datamodule.train_dataset.input_lang.n_words,
                cfg.gru_text.encoder_hidden_size,
                cfg.gru_text.decoder_hidden_size,
                datamodule.train_dataset.output_lang.n_words,
                teacher_forcing=cfg.gru_text.teacher_forcing,
                decoder_dropout_p=cfg.gru_text.decoder_dropout_p,
                learning_rate=cfg.lr,
            )
    raise ValueError(f"Unknown model type {cfg.model_type}")


def get_datamodule(cfg: DictConfig):
    if cfg.model_type == 'naive':
        return NaiveDataModule(
            cfg.data_folder_path,
            cfg.naive.wv2_path,
            cfg.batch_size,
            x_text_pad_length=cfg.naive.x_text_pad_length,
            x_prev_action_pad_length=cfg.naive.x_prev_action_pad_length,
            use_small_dataset=cfg.use_small_dataset,
        )
    if cfg.model_type == 'gru_text':
        return SequentialDataModule(
            cfg.data_folder_path,
            cfg.batch_size,
            input_lang_path=cfg.gru_text.input_lang_path,
            output_lang_path=cfg.gru_text.output_lang_path,
            include_x_text=True,
            include_x_cur_image=True,
            include_x_prev_actions=True,
            use_small_dataset=cfg.use_small_dataset,
        )
    raise ValueError(f"Unknown model type {cfg.model_type}")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Model type: {cfg.model_type}")
    logger.info(f"loading from path: {cfg.data_folder_path}")
    logger.info(f"Using gensim embeddings from {cfg.naive.wv2_path}")
    logger.info(f"Using input/output langs at {cfg.gru_text.input_lang_path}/{cfg.gru_text.output_lang_path}")
    logger.info(f"Saving/loading model checkpoints to/from {cfg.model_checkpoints_path} (model_load_name: {cfg.model_load_name})")

    datamodule = get_datamodule(cfg)
    datamodule.setup("fit")
    datamodule.setup("validate")
    logger.info("train and valid have been setup")

    # create/load model
    model = load_or_create_model(cfg, datamodule)
    logger.info("model loaded")

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model_checkpoints_path,
        save_top_k=cfg.trainer.checkpoints_save_top_k,
        monitor="val_loss"
    )
    trainer = Trainer(
        accelerator=cfg.trainer.acc_device,
        gpus=[0],
        auto_lr_find=cfg.trainer.auto_lr_find,
        track_grad_norm=cfg.trainer.track_grad_norm,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.trainer.max_epochs,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        detect_anomaly=cfg.trainer.detect_anomaly,
    )
    logger.info("trainer created")

    logger.info("Tuning training hyperparameters")
    trainer.tune(model, datamodule=datamodule)
    logger.info(f"Trainer tuned. LR: {model.learning_rate}")

    logger.info("Fitting model...")
    if not cfg.trainer.auto_lr_find:
        model.learning_rate = cfg.lr
    trainer.fit(
        model=model,
        datamodule=datamodule
    )

    logger.info(f"Done! Best model at {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
