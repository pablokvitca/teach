import json
import logging
import os
import re
from collections import Counter

import unicodedata
from typing import Optional, Tuple, List

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

from teach.logger import create_logger
from teach.modeling.toast.Lang import Lang
from transformers import AutoTokenizer, DataCollatorWithPadding

logger = create_logger(__name__, level=logging.INFO)


class TaskFromDialogueHistoryTEACHDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            split_name: str,
            tokenizer,
            num_labels=-1,
            class_id_2_task_id=None,
            task_id_2_class_id=None,
            use_commander_language=True,
            use_follower_language=True,
            use_edh=True,
            use_main_task_only=True,  # TODO: implement not case
    ):
        self.data_dir = data_dir
        self.split_name = split_name

        self._num_labels = num_labels

        self.use_commander_language = use_commander_language
        self.use_follower_language = use_follower_language

        self.use_edh = use_edh

        self.use_main_task_only = use_main_task_only

        self.tokenizer = tokenizer

        self.known_tasks = Counter()
        self.task_id_2_name = {}
        self.task_id_2_class_id = task_id_2_class_id if task_id_2_class_id is not None else {}
        self.class_id_2_task_id = class_id_2_task_id if class_id_2_task_id is not None else {}

        self.data = self._load_data()

    @staticmethod
    def normalize_string(s):
        def unicode_to_ascii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )

        s = unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def num_labels(self):
        return len(self.known_tasks) if self._num_labels == -1 else self._num_labels

    def tokenize_input_language(self, text):
        return self.tokenizer(text, truncation=True, padding=True)

    def get_text_from_instance(self, edh_instance):
        text_parts = []
        cleaned_dialog = edh_instance["dialog_history_cleaned"]
        for dialog_part in cleaned_dialog:
            speaker, utterance = dialog_part
            if speaker == "Commander" and self.use_commander_language:
                text_parts.append(f"{speaker}: {utterance}")
            elif speaker == "Driver" and self.use_follower_language:
                text_parts.append(f"{speaker}: {utterance}")
        return ". ".join(text_parts).replace("..", ".")

    def _load_data(self):
        if self.use_edh:
            return self._load_data_edh()
        else:
            return self._load_data_games()

    def _load_data_games(self):
        edh_dir = os.path.join(self.data_dir, 'edh_instances', self.split_name)
        games_dir = os.path.join(self.data_dir, 'games', self.split_name)
        files = sorted([f for f in os.listdir(edh_dir) if not f.startswith('.')])
        data = {}
        for i in trange(len(files)):
            file = files[i]
            edh_instance_file_path = os.path.join(edh_dir, file)
            with open(edh_instance_file_path) as f:
                edh_instance = json.load(f)
                # CHECK GAME WITH ID FILE EXISTS!
                game_id = edh_instance["game_id"]
                game_file_path = os.path.join(games_dir, f'{game_id}.game.json')
                if os.path.exists(game_file_path):
                    text_from_instance = self.get_text_from_instance(edh_instance)
                    instance_text_tensor = self.tokenize_input_language(text_from_instance)
                    x = instance_text_tensor

                    if game_id in data:
                        data[game_id] = (instance_text_tensor, data[game_id][1], text_from_instance)
                    else:
                        game_tasks: List[Tuple[int, str, str]] = []
                        with open(game_file_path) as game_file:
                            game = json.load(game_file)
                            task_data = game["tasks"][0]
                            if self.use_main_task_only:
                                game_tasks = [(task_data["task_id"], task_data["task_name"], task_data["desc"])]
                            else:
                                game_tasks = TaskFromDialogueHistoryTEACHDataset.recursively_get_task_data(task_data)

                        y = []
                        for task_id, task_name, task_desc in game_tasks:
                            self.known_tasks[task_id] += 1
                            if task_id not in self.task_id_2_class_id:
                                task_name = task_data["task_name"]
                                self.task_id_2_name[task_id] = task_name
                                class_id = len(self.task_id_2_class_id)
                                self.task_id_2_class_id[task_id] = class_id
                                self.class_id_2_task_id[class_id] = task_id
                                logger.info(f"Added new task (id: {task_id}) '{task_name}' as class #{class_id}")
                            y.append(self.task_id_2_class_id[task_id])

                        data[game_id] = (x, y, text_from_instance)
                else:
                    logger.warn(f"GAME FILE FOR EDH INSTANCE DID NOT EXIST \n\tgame: {game_file_path} \n\tedh_instance:{edh_instance_file_path}")
        return list(data.values())

    def _load_data_edh(self):
        edh_dir = os.path.join(self.data_dir, 'edh_instances', self.split_name)
        games_dir = os.path.join(self.data_dir, 'games', self.split_name)
        files = sorted([f for f in os.listdir(edh_dir) if not f.startswith('.')])
        data = []
        for i in trange(len(files)):
            file = files[i]
            edh_instance_file_path = os.path.join(edh_dir, file)
            with open(edh_instance_file_path) as f:
                edh_instance = json.load(f)
                # CHECK GAME WITH ID FILE EXISTS!
                game_id = edh_instance["game_id"]
                game_file_path = os.path.join(games_dir, f'{game_id}.game.json')
                if os.path.exists(game_file_path):
                    text_from_instance = self.get_text_from_instance(edh_instance)
                    instance_text_tensor = self.tokenize_input_language(text_from_instance)
                    x = instance_text_tensor

                    game_tasks: List[Tuple[int, str, str]] = []
                    with open(game_file_path) as game_file:
                        game = json.load(game_file)
                        task_data = game["tasks"][0]
                        if self.use_main_task_only:
                            game_tasks = [(task_data["task_id"], task_data["task_name"], task_data["desc"])]
                        else:
                            game_tasks = TaskFromDialogueHistoryTEACHDataset.recursively_get_task_data(task_data)

                    y = []
                    for task_id, task_name, task_desc in game_tasks:
                        self.known_tasks[task_id] += 1
                        if task_id not in self.task_id_2_class_id:
                            task_name = task_data["task_name"]
                            self.task_id_2_name[task_id] = task_name
                            class_id = len(self.task_id_2_class_id)
                            self.task_id_2_class_id[task_id] = class_id
                            self.class_id_2_task_id[class_id] = task_id
                            logger.info(f"Added new task (id: {task_id}) '{task_name}' as class #{class_id}")
                        y.append(self.task_id_2_class_id[task_id])

                    data.append((x, y, text_from_instance))
                else:
                    logger.warn(f"GAME FILE FOR EDH INSTANCE DID NOT EXIST \n\tgame: {game_file_path} \n\tedh_instance:{edh_instance_file_path}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x, y, original_text = self.data[idx]
        return {
            **x,
            "labels": y,
            "_used_text_data": original_text,
        }

    @staticmethod
    def recursively_get_task_data(task_data) -> List[Tuple[int, str, str]]:
        data = [(task_data["task_id"], task_data["task_name"], task_data["desc"])]
        for component in task_data["components"]:
            if "task" in component.keys():
                component_task_data = component["task"]
                data.extend(TaskFromDialogueHistoryTEACHDataset.recursively_get_task_data(component_task_data))
        return data


class TaskFromDialogueHistoryDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str,
            pretrained_transformer_name="distilbert-base-uncased",
            use_commander_language: bool = True,
            use_follower_language: bool = True,
            use_main_task_only=True,
            use_edh=True,
            insert_pad_token=False,
            use_small_dataset: bool = False,
            train_batch_size: int = 16,
            eval_batch_size: int = 16,  # not used currently
            num_workers: int = 8,
     ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.use_commander_language = use_commander_language
        self.use_follower_language = use_follower_language
        self.use_main_task_only = use_main_task_only
        self.use_edh = use_edh

        self.use_small_dataset = use_small_dataset

        self.train_dataset = None
        self.valid_seen_dataset = None
        self.valid_unseen_dataset = None
        self.test_seen_dataset = None
        self.test_unseen_dataset = None

        self.shared_input_lang: Optional[Lang] = None

        self.pretrained_transformer_name = pretrained_transformer_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_transformer_name, use_fast=True)
        self.insert_pad_token = insert_pad_token
        if insert_pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.num_labels = -1
        self.class_id_2_task_id = None
        self.task_id_2_class_id = None

        self.num_workers = num_workers

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.pretrained_transformer_name, use_fast=True)

    def load_dataset(self, split_name) -> TaskFromDialogueHistoryTEACHDataset:
        dataset = TaskFromDialogueHistoryTEACHDataset(
            self.data_dir,
            split_name,
            self.tokenizer,
            num_labels=self.num_labels,
            class_id_2_task_id=self.class_id_2_task_id,
            task_id_2_class_id=self.task_id_2_class_id,
            use_commander_language=self.use_commander_language,
            use_follower_language=self.use_follower_language,
            use_main_task_only=self.use_main_task_only,
            use_edh=self.use_edh,
        )
        return dataset

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")
        if (stage in ["train", "fit"] or stage is None) and self.train_dataset is None:
            split_name = 'train' if not self.use_small_dataset else 'train_small'
            self.train_dataset = self.load_dataset(split_name)
            self.num_labels = max(self.num_labels, self.train_dataset.num_labels())
            self.class_id_2_task_id = self.train_dataset.class_id_2_task_id
            self.task_id_2_class_id = self.train_dataset.task_id_2_class_id
        if (stage in ["val", "valid", "validate"] or stage is None) and self.valid_seen_dataset is None:
            self.valid_seen_dataset = self.load_dataset('valid_seen')
            self.num_labels = max(self.num_labels, self.valid_seen_dataset.num_labels())
        if (stage in ["val_unseen", "valid_unseen", "validate_unseen"] or stage is None) and self.valid_unseen_dataset is None:
            self.valid_unseen_dataset = self.load_dataset('valid_unseen')
            self.num_labels = max(self.num_labels, self.valid_unseen_dataset.num_labels())
        if (stage == "test" or stage is None) and self.test_seen_dataset is None:
            self.test_seen_dataset = self.load_dataset('test_seen')
            self.num_labels = max(self.num_labels, self.test_seen_dataset.num_labels())
        if (stage == "test_unseen" or stage is None) and self.test_unseen_dataset is None:
            self.test_unseen_dataset = self.load_dataset('test_unseen')
            self.num_labels = max(self.num_labels, self.test_unseen_dataset.num_labels())

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train dataset is not loaded")
        return self._get_dataloader(self.train_dataset)

    def val_dataloader(self):
        if self.valid_seen_dataset is None:
            raise ValueError("valid seen dataset is not loaded")
        return self._get_dataloader(self.valid_seen_dataset)

    def val_unseen_dataloader(self):
        if self.valid_unseen_dataset is None:
            raise ValueError("valid unseen dataset is not loaded")
        return self._get_dataloader(self.valid_unseen_dataset)

    def test_dataloader(self):
        if self.test_seen_dataset is None:
            raise ValueError("test seen dataset is not loaded")
        return self._get_dataloader(self.test_seen_dataset)

    def test_unseen_dataloader(self):
        if self.test_unseen_dataset is None:
            raise ValueError("test unseen dataset is not loaded")
        return self._get_dataloader(self.test_unseen_dataset)
