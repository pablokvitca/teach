import logging
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Union

from torch.cuda import is_available

from teach.inference.actions import all_agent_actions
from teach.logger import create_logger
from teach.modeling.toast.GRUTextOnlyModel import GRUTextOnlyModel
from teach.modeling.toast.NaiveDataModule import NaiveDataModule
from teach.modeling.toast.NaiveMultimodalModel import NaiveMultiModalModel
from teach.modeling.toast.SequentialDataModule import SequentialDataModule
from teach.modeling.toast.SequentialSubgoalDataModule import SequentialSubgoalDataModule

from pytorch_lightning.loggers import WandbLogger

from teach.modeling.toast.TaskFromDialogueHistoryDataModule import TaskFromDialogueHistoryDataModule
from teach.modeling.toast.TextClassificationModel import TextClassificationModel
from teach.modeling.toast.TextMultiLabelClassificationModel import TextMultiLabelClassificationModel
from teach.modeling.toast.utils import flatten_cfg_dict

logger = create_logger(__name__, level=logging.INFO)


def does_model_exist(model_load_path):
    return os.path.exists(model_load_path)


def load_or_create_model(cfg: DictConfig, datamodule: LightningDataModule):
    path = os.path.join(cfg.model_checkpoints_pre_path or '', cfg.model_type, cfg.model_load_name)
    if not cfg.skip_model_load and cfg.model_checkpoints_pre_path is not None and does_model_exist(path):
        logger.info(f"Loading model from {path}.")
        if cfg.model_type == 'naive':
            return NaiveMultiModalModel.load_from_checkpoint(path)
        if cfg.model_type in ['gru_text', 'gru_text_subgoal']:
            return GRUTextOnlyModel.load_from_checkpoint(path)
        if cfg.model_type == 'task_from_text_single':
            return TextClassificationModel.load_from_checkpoint(path)
        if cfg.model_type == 'task_from_text_single_game':
            return TextClassificationModel.load_from_checkpoint(path)
        if cfg.model_type == 'task_from_text_multi':
            return TextMultiLabelClassificationModel.load_from_checkpoint(path)
        if cfg.model_type == 'task_from_text_multi_ame':
            return TextMultiLabelClassificationModel.load_from_checkpoint(path)
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
        if cfg.model_type in ['gru_text', 'gru_text_subgoal']:
            cfg_gru = cfg[cfg.model_type]
            return GRUTextOnlyModel(
                datamodule.train_dataset.input_lang,
                datamodule.train_dataset.output_lang,
                cfg_gru.encoder_hidden_size,
                cfg_gru.decoder_hidden_size,
                teacher_forcing=cfg_gru.teacher_forcing,
                decoder_dropout_p=cfg_gru.decoder_dropout_p,
                learning_rate=cfg.trainer.lr,
                max_length=cfg_gru.max_length,
                max_output_length=cfg_gru.max_output_length,
                output_length_delta=cfg_gru.output_length_delta,
                use_single_optimizer=cfg_gru.use_single_optimizer,
            )
        if cfg.model_type == 'task_from_text_single':
            logger.info(f"Using HuggingFace Pre-Trained transformer {cfg.task_from_text_single.pretrained_model_name}")
            return TextClassificationModel(
                cfg.task_from_text_single.pretrained_model_name,
                num_labels=datamodule.num_labels,
                learning_rate=cfg.task_from_text_single.learning_rate,
                adam_epsilon=cfg.task_from_text_single.adam_epsilon,
                warmup_steps=cfg.task_from_text_single.warmup_steps,
                weight_decay=cfg.task_from_text_single.weight_decay,
                train_batch_size=cfg.datamodule.batch_size,
                eval_batch_size=cfg.datamodule.batch_size,
                freeze_encoder=cfg.task_from_text_single.freeze_encoder,
            )
        if cfg.model_type == 'task_from_text_multi':
            logger.info(f"Using HuggingFace Pre-Trained transformer {cfg.task_from_text_multi.pretrained_model_name}")
            return TextMultiLabelClassificationModel(
                cfg.task_from_text_multi.pretrained_model_name,
                num_labels=datamodule.num_labels,
                learning_rate=cfg.task_from_text_multi.learning_rate,
                adam_epsilon=cfg.task_from_text_multi.adam_epsilon,
                warmup_steps=cfg.task_from_text_multi.warmup_steps,
                weight_decay=cfg.task_from_text_multi.weight_decay,
                train_batch_size=cfg.datamodule.batch_size,
                eval_batch_size=cfg.datamodule.batch_size,
                freeze_encoder=cfg.task_from_text_multi.freeze_encoder,
                label_threshold=cfg.task_from_text_multi.label_threshold,
            )
        if cfg.model_type == 'task_from_text_single_game':
            logger.info(f"Using HuggingFace Pre-Trained transformer {cfg.task_from_text_single.pretrained_model_name}")
            return TextClassificationModel(
                cfg.task_from_text_single_game.pretrained_model_name,
                num_labels=datamodule.num_labels,
                learning_rate=cfg.task_from_text_single_game.learning_rate,
                adam_epsilon=cfg.task_from_text_single_game.adam_epsilon,
                warmup_steps=cfg.task_from_text_single_game.warmup_steps,
                weight_decay=cfg.task_from_text_single_game.weight_decay,
                train_batch_size=cfg.datamodule.batch_size,
                eval_batch_size=cfg.datamodule.batch_size,
                freeze_encoder=cfg.task_from_text_single_game.freeze_encoder,
            )
        if cfg.model_type == 'task_from_text_multi_game':
            logger.info(f"Using HuggingFace Pre-Trained transformer {cfg.task_from_text_multi_game.pretrained_model_name}")
            return TextMultiLabelClassificationModel(
                cfg.task_from_text_multi_game.pretrained_model_name,
                num_labels=datamodule.num_labels,
                learning_rate=cfg.task_from_text_multi_game.learning_rate,
                adam_epsilon=cfg.task_from_text_multi_game.adam_epsilon,
                warmup_steps=cfg.task_from_text_multi_game.warmup_steps,
                weight_decay=cfg.task_from_text_multi_game.weight_decay,
                train_batch_size=cfg.datamodule.batch_size,
                eval_batch_size=cfg.datamodule.batch_size,
                freeze_encoder=cfg.task_from_text_multi_game.freeze_encoder,
                label_threshold=cfg.task_from_text_multi_game.label_threshold,
            )
    raise ValueError(f"Unknown model type {cfg.model_type}")


def get_datamodule(cfg: DictConfig):
    if cfg.model_type == 'naive':
        logger.info(f"Using gensim embeddings from {cfg.naive.wv2_path}")
        return NaiveDataModule(
            cfg.data_folder_path,
            cfg.naive.wv2_path,
            cfg.datamodule.batch_size,
            x_text_pad_length=cfg.naive.x_text_pad_length,
            x_prev_action_pad_length=cfg.naive.x_prev_action_pad_length,
            use_small_dataset=cfg.datamodule.use_small_dataset,
            num_workers=cfg.datamodule.num_workers,
        )
    if cfg.model_type == 'gru_text':
        logger.info(f"Using input/output langs at {cfg.gru_text.input_lang_path}/{cfg.gru_text.output_lang_path}")
        return SequentialDataModule(
            cfg.data_folder_path,
            cfg.datamodule.batch_size,
            validation_batch_size=cfg.datamodule.validation_batch_size,
            input_lang_path=cfg.gru_text.input_lang_path,
            output_lang_path=cfg.gru_text.output_lang_path,
            include_x_text=True,
            include_x_cur_image=False,
            include_x_prev_actions=True,
            use_small_dataset=cfg.datamodule.use_small_dataset,
            num_workers=cfg.datamodule.num_workers,
        )
    if cfg.model_type == 'gru_text_subgoal':
        logger.info(f"Using input/output langs at {cfg.gru_text_subgoal.input_lang_path}/{cfg.gru_text_subgoal.output_lang_path}")
        return SequentialSubgoalDataModule(
            cfg.data_folder_path,
            cfg.datamodule.batch_size,
            validation_batch_size=cfg.datamodule.validation_batch_size,
            input_lang_path=cfg.gru_text_subgoal.input_lang_path,
            output_lang_path=cfg.gru_text_subgoal.output_lang_path,
            include_x_text=True,
            use_subgoal_history=cfg.gru_text_subgoal.use_subgoal_history,
            use_subgoal_future=cfg.gru_text_subgoal.use_subgoal_future,
            use_commander_language=cfg.gru_text_subgoal.use_commander_language,
            use_follower_language=cfg.gru_text_subgoal.use_follower_language,
            use_small_dataset=cfg.datamodule.use_small_dataset,
            num_workers=cfg.datamodule.num_workers,
        )
    if cfg.model_type == 'task_from_text_single':
        logger.info(f"Using HuggingFace Pre-Trained transformer {cfg.task_from_text_single.pretrained_model_name}")
        return TaskFromDialogueHistoryDataModule(
            cfg.data_folder_path,
            pretrained_transformer_name=cfg.task_from_text_single.pretrained_model_name,
            use_commander_language=cfg.task_from_text_single.use_commander_language,
            use_follower_language=cfg.task_from_text_single.use_follower_language,
            use_main_task_only=True,
            use_edh=True,
            insert_pad_token=cfg.task_from_text_single.insert_pad_token,
            use_small_dataset=cfg.datamodule.use_small_dataset,
            train_batch_size=cfg.datamodule.batch_size,
            eval_batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
        )
    if cfg.model_type == 'task_from_text_multi':
        logger.info(f"Using HuggingFace Pre-Trained transformer {cfg.task_from_text_multi.pretrained_model_name}")
        return TaskFromDialogueHistoryDataModule(
            cfg.data_folder_path,
            pretrained_transformer_name=cfg.task_from_text_multi.pretrained_model_name,
            use_commander_language=cfg.task_from_text_multi.use_commander_language,
            use_follower_language=cfg.task_from_text_multi.use_follower_language,
            use_main_task_only=False,
            use_edh=True,
            insert_pad_token=cfg.task_from_text_multi.insert_pad_token,
            use_small_dataset=cfg.datamodule.use_small_dataset,
            train_batch_size=cfg.datamodule.batch_size,
            eval_batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
        )
    if cfg.model_type == 'task_from_text_single_game':
        logger.info(f"Using HuggingFace Pre-Trained transformer {cfg.task_from_text_single_game.pretrained_model_name}")
        return TaskFromDialogueHistoryDataModule(
            cfg.data_folder_path,
            pretrained_transformer_name=cfg.task_from_text_single_game.pretrained_model_name,
            use_commander_language=cfg.task_from_text_single_game.use_commander_language,
            use_follower_language=cfg.task_from_text_single_game.use_follower_language,
            use_main_task_only=True,
            use_edh=False,
            insert_pad_token=cfg.task_from_text_single_game.insert_pad_token,
            use_small_dataset=cfg.datamodule.use_small_dataset,
            train_batch_size=cfg.datamodule.batch_size,
            eval_batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
        )
    if cfg.model_type == 'task_from_text_multi_game':
        logger.info(f"Using HuggingFace Pre-Trained transformer {cfg.task_from_text_multi_game.pretrained_model_name}")
        return TaskFromDialogueHistoryDataModule(
            cfg.data_folder_path,
            pretrained_transformer_name=cfg.task_from_text_multi_game.pretrained_model_name,
            use_commander_language=cfg.task_from_text_multi_game.use_commander_language,
            use_follower_language=cfg.task_from_text_multi_game.use_follower_language,
            use_main_task_only=False,
            use_edh=False,
            insert_pad_token=cfg.task_from_text_multi_game.insert_pad_token,
            use_small_dataset=cfg.datamodule.use_small_dataset,
            train_batch_size=cfg.datamodule.batch_size,
            eval_batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.num_workers,
        )
    raise ValueError(f"Unknown model type {cfg.model_type}")


def save_data_preprocessing(cfg: DictConfig, datamodule: LightningDataModule):
    if cfg.model_type == 'naive':
        logger.info("No need to save data preprocessing for naive model")
    if cfg.model_type in ['gru_text', 'gru_text_subgoal']:
        logger.info(f"Saving data preprocessing for {cfg.model_type}")
        cfg_gru = cfg[cfg.model_type]
        assert isinstance(datamodule, SequentialDataModule) or isinstance(datamodule, SequentialSubgoalDataModule), \
            f"Expected SequentialDataModule or SequentialSubgoalDataModule, got {type(datamodule)}"
        sequential_datamodule: Union[SequentialDataModule, SequentialSubgoalDataModule] = datamodule
        if cfg_gru.save_input_lang_path:
            logger.info(f"Saving input lang at {cfg_gru.save_input_lang_path}")
            sequential_datamodule.shared_input_lang.save(cfg_gru.save_input_lang_path)
        if cfg_gru.save_output_lang_path:
            logger.info(f"Saving output lang at {cfg_gru.save_output_lang_path}")
            sequential_datamodule.shared_output_lang.save(cfg_gru.save_output_lang_path)
    if cfg.model_type == 'task_from_text_single':
        logger.info("No need to save data preprocessing for task_from_text_single model")
    if cfg.model_type == 'task_from_text_multi':
        logger.info("No need to save data preprocessing for task_from_text_single model")
    if cfg.model_type == 'task_from_text_single_game':
        logger.info("No need to save data preprocessing for task_from_text_single_game model")
    if cfg.model_type == 'task_from_text_multi_game':
        logger.info("No need to save data preprocessing for task_from_text_single_game model")


def check_loaded_datamodule(cfg: DictConfig, datamodule: LightningDataModule):
    if cfg.model_type == 'naive':
        pass
    if cfg.model_type in ['gru_text', 'gru_text_subgoal']:
        cfg_gru = cfg[cfg.model_type]
        if cfg.datamodule.fail_if_cannot_load:
            if (cfg_gru.input_lang_path is not None and cfg_gru.input_lang_path != 'None') \
                    and not datamodule.shared_input_lang.loaded_from_file:
                raise ValueError(f"Could not load input langs from {cfg_gru.input_lang_path}")
            if (cfg_gru.output_lang_path is not None and cfg_gru.output_lang_path != 'None') \
                    and not datamodule.shared_output_lang.loaded_from_file:
                raise ValueError(f"Could not load output langs from {cfg_gru.output_lang_path}")
    if cfg.model_type == 'task_from_text_single':
        pass
    if cfg.model_type == 'task_from_text_multi':
        pass
    if cfg.model_type == 'task_from_text_single_game':
        pass
    if cfg.model_type == 'task_from_text_multi_game':
        pass


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    wandb_logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        offline=cfg.wandb.offline,
        group=cfg.wandb.group_name,
        log_model=cfg.wandb.log_model,
        config=cfg,
    )
    wandb_logger.log_hyperparams(flatten_cfg_dict(cfg))

    if cfg.model_type not in cfg.known_model_data_types:
        raise ValueError(f"Unknown model type {cfg.model_type}")
    logger.info(f"Model type {cfg.model_type}")
    logger.info(f"loading from path {cfg.data_folder_path}")
    logger.info(f"Saving/loading model checkpoints to/from {cfg.model_checkpoints_pre_path} (model_load_name {cfg.model_load_name})")

    datamodule = get_datamodule(cfg)
    datamodule.setup("fit")
    datamodule.setup("validate")
    logger.info("train and valid have been setup")

    # Checking loaded datamodule
    check_loaded_datamodule(cfg, datamodule)

    if cfg.datamodule.save_data_preprocessing:
        save_data_preprocessing(cfg, datamodule)
        logger.info("data preprocessing saved")

    # create/load model
    model = load_or_create_model(cfg, datamodule)
    logger.info("model loaded")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.model_checkpoints_pre_path, cfg.model_type),
        save_top_k=cfg.trainer.checkpoints_save_top_k,
        monitor="val_loss"
    )

    devices = cfg.trainer.devices
    if cfg.trainer.acc_device == 'gpu':
        if is_available():
            if isinstance(devices, str):
                devices = [int(d) for d in devices.split(',')]
        else:
            devices = cfg.trainer.fallback_devices

    trainer = Trainer(
        accelerator=cfg.trainer.acc_device if is_available() else 'cpu',
        devices=devices,
        auto_lr_find=cfg.trainer.auto_lr_find and cfg[cfg.model_type].auto_lr_find,
        track_grad_norm=cfg.trainer.track_grad_norm,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.trainer.max_epochs,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        detect_anomaly=cfg.trainer.detect_anomaly,
        fast_dev_run=cfg.trainer.fast_dev_run,
        # overfit_batches=1,
        # check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        # val_check_interval=cfg.trainer.val_check_interval,
        logger=wandb_logger
    )
    logger.info("trainer created")

    if cfg.trainer.auto_lr_find or cfg.trainer.auto_batch_find:
        logger.info("Tuning training hyper parameters")
        trainer.tune(model, datamodule=datamodule)
        logger.info(f"Trainer tuned. LR: {model.learning_rate}")
    else:
        logger.info("Skipped tuning")

    logger.info("Fitting model...")
    if not cfg.trainer.auto_lr_find and not cfg.trainer.use_model_lr:
        model.learning_rate = cfg.trainer.lr
    trainer.fit(
        model=model,
        datamodule=datamodule
    )

    logger.info(f"Done! Best model at {checkpoint_callback.best_model_path}")

    best_model = model.__class__.load_from_checkpoint(checkpoint_callback.best_model_path)

    datamodule.setup("val")
    datamodule.setup("val_unseen")
    logger.info("Testing...")
    trainer.test(
        best_model,
        dataloaders=datamodule.val_unseen_dataloader()
    )

    wandb_logger.save()
    wandb_logger.finalize("done")
    wandb_logger.close()


if __name__ == "__main__":
    main()
