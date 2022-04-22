import json
import logging
import os
import re
import unicodedata
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

from teach.logger import create_logger
from teach.modeling.toast.Lang import Lang

logger = create_logger(__name__, level=logging.INFO)


class SequentialTEACHSubgoalDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            split_name: str,
            include_x_test: bool,
            input_lang_path=None,
            output_lang_path=None,
            input_lang=None,
            output_lang=None,
            token_pad_length=300,
            extend_language=True,
            use_subgoal_history=True,
            use_subgoal_future=True,
            use_commander_language=True,
            use_follower_language=True,
    ):
        self.data_dir = data_dir
        self.split_name = split_name
        self.include_x_text = include_x_test
        self.token_pad_length = token_pad_length

        self.input_lang_path = input_lang_path if os.path.exists(input_lang_path) else None
        self.input_lang = input_lang or Lang(self.input_lang_path)
        self.output_lang_path = output_lang_path if os.path.exists(output_lang_path) else None
        self.output_lang = output_lang or Lang(self.output_lang_path)

        self.extend_language = extend_language

        self.use_subgoal_history = use_subgoal_history
        self.use_subgoal_future = use_subgoal_future

        self.use_commander_language = use_commander_language
        self.use_follower_language = use_follower_language

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

    @staticmethod
    def _tensor_from_sentence(lang, token_list):
        indexes = [lang.word2index[word] for word in token_list]
        indexes.append(lang.EOS_token_index)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

    def tensorize_input_language(self, token_list):
        return SequentialTEACHSubgoalDataset._tensor_from_sentence(self.input_lang, token_list)

    def tensorize_subgoal_language(self, token_list):
        return SequentialTEACHSubgoalDataset._tensor_from_sentence(self.output_lang, token_list)

    def get_text_tokens_from_instance(self, edh_instance):
        tokens_list = []
        cleaned_dialog = edh_instance["dialog_history_cleaned"]
        for dialog_part in cleaned_dialog:
            speaker, utterance = dialog_part
            if speaker == "Commander" and self.use_commander_language:
                tokens_list.extend(SequentialTEACHSubgoalDataset.normalize_string(utterance).split(" "))
            elif speaker == "Driver" and self.use_follower_language:
                tokens_list.extend(SequentialTEACHSubgoalDataset.normalize_string(utterance).split(" "))
        return tokens_list

    def _load_data(self):
        edh_dir = os.path.join(self.data_dir, 'edh_instances', self.split_name)
        files = sorted(os.listdir(edh_dir))
        data = []
        for i in trange(len(files)):
            file = files[i]
            with open(os.path.join(edh_dir, file)) as f:
                edh_instance = json.load(f)
                if self.include_x_text:
                    text_from_instance = SequentialTEACHSubgoalDataset.get_text_tokens_from_instance(edh_instance)
                    if self.input_lang_path is None and self.extend_language:
                        [self.input_lang.add_word(word) for word in text_from_instance]
                    instance_text_tensor = self.tensorize_input_language(text_from_instance)

                history_subgoals, future_subgoals = edh_instance["history_subgoals"], edh_instance["future_subgoals"]
                subgoals = (history_subgoals if self.use_subgoal_history else []) + \
                           (future_subgoals if self.use_subgoal_future else [])

                if self.output_lang_path is None:
                    logger.error("SUBGOAL LANGUAGE SHOULD BE PRELOADED!")
                    if self.extend_language:
                        [self.output_lang.add_word(subgoal) for subgoal in subgoal]

                instance_subgoal_tensor = self.tensorize_subgoal_language(subgoals)

                x = instance_text_tensor
                y = instance_subgoal_tensor

                data.append((x, y))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x, y = self.data[idx]
        return x, y


class SequentialSubgoalDataModule(LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 input_lang_path=None,
                 output_lang_path=None,
                 include_x_text: bool = True,
                 use_subgoal_history: bool = True,
                 use_subgoal_future: bool = True,
                 use_commander_language: bool = True,
                 use_follower_language: bool = True,
                 use_small_dataset: bool = False,
                 num_workers: int = 8,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_lang_path = input_lang_path
        self.output_lang_path = output_lang_path
        self.include_x_text = include_x_text
        self.use_subgoal_history = use_subgoal_history
        self.use_subgoal_future = use_subgoal_future
        self.use_commander_language = use_commander_language
        self.use_follower_language = use_follower_language

        self.use_small_dataset = use_small_dataset

        self.train_dataset = None
        self.valid_seen_dataset = None
        self.valid_unseen_dataset = None
        self.test_seen_dataset = None
        self.test_unseen_dataset = None

        self.shared_input_lang = None
        self.shared_output_lang = None

        self.num_workers = num_workers

    @staticmethod
    def collate_fn_pad(batch):
        x, y = zip(*batch)

        # lengths
        x_lengths = Tensor([t.shape[0] for t in x])
        y_lengths = Tensor([t.shape[0] for t in y])

        # pad
        x = pad_sequence(x)
        y = pad_sequence(y)

        # compute mask
        x_mask = (x != 0)
        y_mask = (y != 0)

        batch = x, y

        return batch, (x_lengths, y_lengths), (x_mask, y_mask)

    def load_dataset(self, split_name, extend_language=False) -> Dataset:
        dataset = SequentialTEACHSubgoalDataset(
            self.data_dir,
            split_name,
            self.include_x_text,
            input_lang=self.shared_input_lang,
            output_lang=self.shared_output_lang,
            extend_language=extend_language,
            use_subgoal_history=self.use_subgoal_history,
            use_subgoal_future=self.use_subgoal_future,
            use_commander_language=self.use_commander_language,
            use_follower_language=self.use_follower_language,
        )
        self.shared_input_lang = dataset.input_lang
        self.shared_output_lang = dataset.output_lang
        return dataset

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")
        if (stage in ["train", "fit"] or stage is None) and self.train_dataset is None:
            split_name = 'train' if not self.use_small_dataset else 'train_small'
            self.train_dataset = self.load_dataset(split_name, extend_language=True)
        if (stage in ["val", "valid", "validate"] or stage is None) and self.valid_seen_dataset is None:
            self.valid_seen_dataset = self.load_dataset('valid_seen', extend_language=False)
        if (stage in ["val_unseen", "valid_unseen", "validate_unseen"] or stage is None) and self.valid_unseen_dataset is None:
            self.valid_unseen_dataset = self.load_dataset('valid_unseen', extend_language=False)
        if (stage == "test" or stage is None) and self.test_seen_dataset is None:
            self.test_seen_dataset = self.load_dataset('test_seen', extend_language=False)
        if (stage == "test_unseen" or stage is None) and self.test_unseen_dataset is None:
            self.test_unseen_dataset = self.load_dataset('test_unseen', extend_language=False)

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=SequentialSubgoalDataModule.collate_fn_pad,
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
