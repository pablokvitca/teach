import json
import logging
import os
from typing import Optional

import torch
from PIL import Image
from gensim.models import KeyedVectors
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

from teach.dataset.definitions import Definitions
from teach.inference.actions import all_agent_actions
from teach.logger import create_logger
from teach.modeling.et.alfred.nn.transforms import Transforms
from teach.modeling.toast.utils import get_text_tokens_from_instance, pad_list, encode_as_word_vectors

logger = create_logger(__name__, level=logging.INFO)


class NaiveTEACHDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            w2v_path: str,
            split_name: str,
            include_x_test: bool,
            include_x_cur_image: bool,
            include_x_prev_actions: bool,
            x_test_seq: bool,
            x_prev_action_seq: bool,
            x_text_pad_length: int,
            x_prev_action_pad_length: int,
    ):
        self.data_dir = data_dir
        self.split_name = split_name
        self.include_x_test = include_x_test
        self.include_x_cur_image = include_x_cur_image
        self.include_x_prev_actions = include_x_prev_actions
        self.x_test_seq = x_test_seq
        self.x_prev_action_seq = x_prev_action_seq
        self.x_text_pad_length = x_text_pad_length
        self.x_prev_action_pad_length = x_prev_action_pad_length

        self.w2v_model = KeyedVectors.load_word2vec_format(
            w2v_path, binary=True, limit=100000
        )

        self.all_agent_actions = set(all_agent_actions)
        definitions = Definitions(version="2.0")
        self._onehot_action_tensors = {}
        for idx, action_name in enumerate(all_agent_actions):
            action_onehot = torch.zeros(len(all_agent_actions))
            action_onehot[idx] = 1
            action_id = definitions.map_actions_name2info[action_name]["action_id"]
            self._onehot_action_tensors[action_id] = action_onehot

        self._img_transform = Transforms.get_transform("default")

        self.data = self._load_data()

    def action_id_to_one_hot(self, action_id):
        return self._onehot_action_tensors[action_id]

    def prev_actions_list_to_matrix(self, prev_actions):
        prev_actions_mat = torch.zeros((len(all_agent_actions), self.x_prev_action_pad_length))

        prev_actions = prev_actions[-self.x_prev_action_pad_length:]

        for idx, prev_action in enumerate(prev_actions):
            prev_actions_mat[:, idx] = prev_action

        return prev_actions_mat

    def tensorize_image(self, img):
        return self._img_transform(img)

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

    def _load_data(self):
        edh_dir = os.path.join(self.data_dir, 'edh_instances', self.split_name)
        files = sorted(os.listdir(edh_dir))
        data = []
        for i in trange(len(files)):
            file = files[i]
            with open(os.path.join(edh_dir, file)) as f:
                edh_instance = json.load(f)
                if self.include_x_test:
                    text_from_instance = get_text_tokens_from_instance(edh_instance)
                    text_from_instance = pad_list(text_from_instance, self.x_text_pad_length)
                    instance_text_encoded = encode_as_word_vectors(self.w2v_model, text_from_instance)

                prev_actions = []
                observed_actions = 0
                action_history, image_history = edh_instance["driver_action_history"], edh_instance["driver_image_history"]
                action_future = edh_instance["driver_actions_future"]
                for idx, (action, img_filename) in enumerate(zip(action_history, image_history)):
                    if action["action_name"] in self.all_agent_actions:
                        next_action = self.get_next_action(action_history, action_future, idx)
                        if next_action is not None:
                            y = self.action_id_to_one_hot(next_action["action_id"])

                            if self.include_x_cur_image:
                                instance_image = os.path.join(edh_instance["game_id"], img_filename)
                            if self.include_x_prev_actions:
                                action_onehot = self.action_id_to_one_hot(action['action_id'])
                                prev_actions.append(action_onehot)
                                observed_actions += 1
                                instance_prev_actions = self.prev_actions_list_to_matrix(prev_actions)

                            x = {
                                "text": instance_text_encoded,
                                "cur_image": instance_image,
                                "prev_actions": instance_prev_actions
                            }

                            data.append((x, y))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x, y = self.data[idx]
        x_cur_img = self.load_img(x["cur_image"])
        return {"text": x["text"], "cur_image": x_cur_img, "prev_actions": x["prev_actions"]}, y


class NaiveDataModule(LightningDataModule):
    """
    This module can be configured to preprocess that 3 modes of data in different ways, this is done *before* feeding
    to an actual model.
    This module is meant to have a "naive" representation of the output, where the sequence nature of the Y is ignored,
    this is meant for testing and as a baseline. Therefore the module loads the episodes from the dataset and splits
    them into separate data instances where each X is a tuple with:
        - x_text, the full text for the episode, vectorized to tensors
        - x_cur_image, the image of the agent before the action is selected, as 3D tensors (width, height, channel)
        - x_prev_actions, the list of previous actions, as tensors
    And where each Y is a single action one hot vector, as a tensor

    Each of the components of X can be disabled, so it is not included on the resulting data, using the arguments:
        - `include_x_text`
        - `include_x_cur_image`
        - `include_x_prev_actions`
    The x_text and x_prev_actions can be configured to be sequences of tensors or a PADDED tensor matrix, using the
    arguments `x_test_seq` and `x_prev_action_seq`. If using a PADDED tensor matrix, the PAD size for each input can be
    configured using the arguments `x_text_pad_length` and `x_prev_action_pad_length`.
    """
    def __init__(self,
                 data_dir: str,
                 wv2_path: str,
                 batch__size: int,
                 include_x_text: bool = True,
                 include_x_cur_image: bool = True,
                 include_x_prev_actions: bool = True,
                 x_test_seq: bool = False,
                 x_prev_action_seq: bool = False,
                 x_text_pad_length: int = 50,
                 x_prev_action_pad_length: int = 500,
                 use_small_dataset: bool = False,
                 num_workers: int = 8,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.wv2_path = wv2_path
        self.batch_size = batch__size
        self.include_x_text = include_x_text
        self.include_x_cur_image = include_x_cur_image
        self.include_x_prev_actions = include_x_prev_actions
        self.x_test_seq = x_test_seq
        self.x_prev_action_seq = x_prev_action_seq
        self.x_text_pad_length = x_text_pad_length
        self.x_prev_action_pad_length = x_prev_action_pad_length

        self.use_small_dataset = use_small_dataset

        self.train_dataset = None
        self.valid_seen_dataset = None
        self.valid_unseen_dataset = None
        self.test_seen_dataset = None
        self.test_unseen_dataset = None

        self.num_workers = num_workers

    def load_dataset(self, split_name) -> Dataset:
        return NaiveTEACHDataset(
            self.data_dir,
            self.wv2_path,
            split_name,
            self.include_x_text,
            self.include_x_cur_image,
            self.include_x_prev_actions,
            self.x_test_seq,
            self.x_prev_action_seq,
            self.x_text_pad_length,
            self.x_prev_action_pad_length
        )

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

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train dataset is not loaded")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.valid_seen_dataset is None:
            raise ValueError("valid seen dataset is not loaded")
        return DataLoader(self.valid_seen_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_unseen_dataloader(self):
        if self.valid_unseen_dataset is None:
            raise ValueError("valid unseen dataset is not loaded")
        return DataLoader(self.valid_unseen_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_seen_dataset is None:
            raise ValueError("test seen dataset is not loaded")
        return DataLoader(self.test_seen_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_unseen_dataloader(self):
        if self.test_unseen_dataset is None:
            raise ValueError("test unseen dataset is not loaded")
        return DataLoader(self.test_unseen_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
