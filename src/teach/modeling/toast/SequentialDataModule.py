import json
import logging
import os
import pickle
import re
import unicodedata
from typing import Optional

import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

from teach.dataset.definitions import Definitions
from teach.inference.actions import all_agent_actions
from teach.logger import create_logger
from teach.modeling.et.alfred.nn.transforms import Transforms

logger = create_logger(__name__, level=logging.INFO)

SOS_token = 0
EOS_token = 1
PAD_token = 2


class Lang:
    def __init__(self, lang_path=None):
        self.word2index = {}
        self.word2count = {}
        self.SOS_token_index, self.SOS_token = 0, '<SOS>'
        self.EOS_token_index, self.EOS_token = 1, '<EOS>'
        self.PAD_token_index, self.PAD_token = 2, '<PAD>'
        self.index2word = {
            self.SOS_token_index: self.SOS_token,
            self.EOS_token_index: self.EOS_token,
            self.PAD_token_index: self.PAD_token
        }
        self.n_words = 3  # Count SOS and EOS
        if lang_path is not None:
            if os.path.exists(lang_path):
                self.load(lang_path)

    def load(self, lang_path):
        _lang = pickle.load(open(lang_path, 'rb'))
        self.word2index = _lang["word2index"]
        self.word2count = _lang["word2count"]
        self.index2word = _lang["index2word"]
        self.n_words = _lang["n_words"]
        self.SOS_token_index = _lang["SOS_token_index"]
        self.SOS_token = _lang["SOS_token"]
        self.EOS_token_index = _lang["EOS_token_index"]
        self.EOS_token = _lang["EOS_token"]
        self.PAD_token_index = _lang["PAD_token_index"]
        self.PAD_token = _lang["PAD_token"]

    def save(self, lang_path):
        pickle.dump({
            "word2index": self.word2index,
            "word2count": self.word2count,
            "index2word": self.index2word,
            "n_words": self.n_words,
            "SOS_token_index": self.SOS_token_index,
            "SOS_token": self.SOS_token,
            "EOS_token_index": self.EOS_token_index,
            "EOS_token": self.EOS_token,
            "PAD_token_index": self.PAD_token_index,
            "PAD_token": self.PAD_token,
        }, open(lang_path, 'wb'))

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word, override_index=None):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words if override_index is None else override_index] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class SequentialTEACHDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            split_name: str,
            include_x_test: bool,
            include_x_cur_image: bool,
            include_x_prev_actions: bool,
            input_lang_path=None,
            output_lang_path=None,
            input_lang=None,
            output_lang=None,
            token_pad_length=300,
    ):
        self.data_dir = data_dir
        self.split_name = split_name
        self.include_x_text = include_x_test
        self.include_x_cur_image = include_x_cur_image
        self.include_x_prev_actions = False and include_x_prev_actions
        self.token_pad_length = token_pad_length

        self.all_agent_actions = set(all_agent_actions)
        definitions = Definitions(version="2.0")
        self._onehot_action_tensors = {}
        for idx, action_name in enumerate(all_agent_actions):
            action_onehot = torch.zeros(len(all_agent_actions))
            action_onehot[idx] = 1
            action_id = definitions.map_actions_name2info[action_name]["action_id"]
            self._onehot_action_tensors[action_id] = action_onehot

        self._img_transform = Transforms.get_transform("default")

        # assert (input_lang_path is not None) != (input_lang is not None)
        # assert (output_lang_path is not None) != (output_lang is not None)

        self.input_lang_path = input_lang_path if os.path.exists(input_lang_path) else None
        self.input_lang = input_lang or Lang(self.input_lang_path)
        self.output_lang_path = output_lang_path if os.path.exists(output_lang_path) else None
        self.output_lang = output_lang or Lang(self.output_lang_path)

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

    def action_id_to_one_hot(self, action_id):
        return self._onehot_action_tensors[action_id]

    def tensorize_image(self, img):
        return self._img_transform(img)

    @staticmethod
    def _tensor_from_sentence(lang, token_list):
        indexes = [lang.word2index[word] for word in token_list]
        indexes.append(lang.EOS_token_index)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

    def tensorize_input_language(self, token_list):
        return SequentialTEACHDataset._tensor_from_sentence(self.input_lang, token_list)

    def tensorize_action_language(self, token_list):
        return SequentialTEACHDataset._tensor_from_sentence(self.output_lang, token_list)

    def load_img(self, path_suffix):
        filepath = os.path.join(self.data_dir, 'images', self.split_name, path_suffix)
        with Image.open(filepath) as img:
            return self.tensorize_image(img)

    def get_next_action(self, action_history, action_future, idx):
        actions = action_history + action_future
        for action in actions[idx+1:]:
            if action["action_name"] in self.all_agent_actions:
                return action
        return None

    @staticmethod
    def get_text_tokens_from_instance(edh_instance):
        tokens_list = []
        cleaned_dialog = edh_instance["dialog_history_cleaned"]
        for dialog_part in cleaned_dialog:
            dialog_part = SequentialTEACHDataset.normalize_string(dialog_part[1])
            tokens_list.extend(dialog_part.split())
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
                    text_from_instance = SequentialTEACHDataset.get_text_tokens_from_instance(edh_instance)
                    if self.input_lang_path is None:
                        [self.input_lang.add_word(word) for word in text_from_instance]
                    instance_text_tensor = self.tensorize_input_language(text_from_instance)

                action_history, image_history = edh_instance["driver_action_history"], edh_instance["driver_image_history"]
                filtered_actions, filtered_images = [], []
                for idx, (action, img_filename) in enumerate(zip(action_history, image_history)):
                    if action["action_name"] in self.all_agent_actions:
                        filtered_actions.append(action)
                        filtered_images.append(img_filename)

                if self.include_x_cur_image:
                    instance_images = [
                        os.path.join(edh_instance["game_id"], img_filename) for img_filename in filtered_images
                    ]

                if self.output_lang_path is None:
                    [self.output_lang.add_word(act["action_name"], override_index=act["action_id"])
                     for act in filtered_actions]
                instance_actions = self.tensorize_action_language([act["action_name"] for act in filtered_actions])

                x = instance_text_tensor
                y = instance_actions

                data.append((x, y))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x, y = self.data[idx]
        # x_cur_img = [self.load_img(img_file) for img_file in x["cur_image"]] if self.include_x_cur_image else None
        return x, y


class SequentialDataModule(LightningDataModule):
    """
    This module can be configured to preprocess that 3 modes of data in different ways, this is done *before* feeding
    to an actual model.
    This module is meant to have a "naive" representation of the output, where the sequence nature of the Y is ignored,
    this is meant for testing and as a baseline. Therefor the module loads the episodes from the dataset and splits
    them into separate data instances where each X is a tuple with:
        - x_text, the full text for the episode, vectorized to tensors
        - x_cur_image, the image of the agent before the action is selected, as 3D tensors (width, height, channel)
        - x_prev_actions, the list of previous actions, as tensors
    And where each Y is a single action one hot vector, as a tensor

    Each of the components of X can be disabled, so it is not included on the resulting data, using the arguments:
        - `include_x_text`
        - `include_x_cur_image`
        - `include_x_prev_actions`

    This uses sequences for text and prev actions.
    """
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 input_lang_path=None,
                 output_lang_path=None,
                 include_x_text: bool = True,
                 include_x_cur_image: bool = True,
                 include_x_prev_actions: bool = True,
                 use_small_dataset: bool = False,
                 num_workers: int = 8,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_lang_path = input_lang_path
        self.output_lang_path = output_lang_path
        self.include_x_text = include_x_text
        self.include_x_cur_image = include_x_cur_image
        self.include_x_prev_actions = include_x_prev_actions

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

    def load_dataset(self, split_name) -> Dataset:
        dataset = SequentialTEACHDataset(
            self.data_dir,
            split_name,
            self.include_x_text,
            self.include_x_cur_image,
            self.include_x_prev_actions,
            input_lang_path=self.input_lang_path,
            output_lang_path=self.output_lang_path,
            input_lang=self.shared_input_lang,
            output_lang=self.shared_output_lang,
        )
        self.shared_input_lang = dataset.input_lang
        self.shared_output_lang = dataset.output_lang
        return dataset

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")
        if (stage in ["train", "fit"] or stage is None) and self.train_dataset is None:
            split_name = 'train' if not self.use_small_dataset else 'train_small'
            self.train_dataset = self.load_dataset(split_name)
        if (stage in ["val", "valid", "validate"] or stage is None) and self.valid_seen_dataset is None:
            self.valid_seen_dataset = self.load_dataset('valid_seen')
        if (stage in ["val_unseen", "valid_unseen", "validate_unseen"] or stage is None) and self.valid_unseen_dataset is None:
            self.valid_unseen_dataset = self.load_dataset('valid_unseen')
        if (stage == "test" or stage is None) and self.test_seen_dataset is None:
            self.test_seen_dataset = self.load_dataset('test_seen')
        if (stage == "test_unseen" or stage is None) and self.test_unseen_dataset is None:
            self.test_unseen_dataset = self.load_dataset('test_unseen')

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=SequentialDataModule.collate_fn_pad,
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
