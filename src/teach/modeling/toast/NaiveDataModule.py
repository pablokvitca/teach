from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class NaiveTEACHDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            include_x_test: bool,
            include_x_cur_image: bool,
            include_x_prev_actions: bool,
            x_test_seq: bool,
            x_prev_action_seq: bool,
            x_text_pad_length: int,
            x_prev_action_pad_length: int,
    ):
        self.data_path = data_path
        self.include_x_test = include_x_test
        self.include_x_cur_image = include_x_cur_image
        self.include_x_prev_actions = include_x_prev_actions
        self.x_test_seq = x_test_seq
        self.x_prev_action_seq = x_prev_action_seq
        self.x_text_pad_length = x_text_pad_length
        self.x_prev_action_pad_length = x_prev_action_pad_length

        self.data = []  # TODO: load data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


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
                 batch__size: int,
                 include_x_text: bool = True,
                 include_x_cur_image: bool = True,
                 include_x_prev_actions: bool = True,
                 x_test_seq: bool = False,
                 x_prev_action_seq: bool = False,
                 x_text_pad_length: int = 50,
                 x_prev_action_pad_length: int = 500,
                 use_small_dataset: bool = False,
                 ):
        super().__init__()
        self.data_dir = data_dir
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

    def load_dataset(self, data_path: str) -> Dataset:
        return NaiveTEACHDataset(
            data_path,
            self.include_x_text,
            self.include_x_cur_image,
            self.include_x_prev_actions,
            self.x_test_seq,
            self.x_prev_action_seq,
            self.x_text_pad_length,
            self.x_prev_action_pad_length
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "train" or stage is None:
            train_data_dir = f"{self.data_dir}/{'train' if not self.use_small_dataset else 'train_small'}"
            self.train_dataset = self.load_dataset(train_data_dir)
        if stage == "valid" or stage is None:
            valid_seen_data_dir = f"{self.data_dir}/valid_seen"
            self.valid_seen_dataset = self.load_dataset(valid_seen_data_dir)
        if stage == "valid_unseen" or stage is None:
            valid_unseen_data_dir = f"{self.data_dir}/valid_unseen"
            self.valid_unseen_dataset = self.load_dataset(valid_unseen_data_dir)
        if stage == "test" or stage is None:
            test_seen_data_dir = f"{self.data_dir}/test_seen"
            self.test_seen_dataset = self.load_dataset(test_seen_data_dir)
        if stage == "test_unseen" or stage is None:
            test_unseen_data_dir = f"{self.data_dir}/test_unseen"
            self.test_unseen_dataset = self.load_dataset(test_unseen_data_dir)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train dataset is not loaded")
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        if self.valid_seen_dataset is None:
            raise ValueError("valid seen dataset is not loaded")
        return DataLoader(self.valid_seen_dataset, batch_size=self.batch_size)

    def val_unseen_dataloader(self):
        if self.valid_unseen_dataset is None:
            raise ValueError("valid unseen dataset is not loaded")
        return DataLoader(self.valid_unseen_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.test_seen_dataset is None:
            raise ValueError("test seen dataset is not loaded")
        return DataLoader(self.test_seen_dataset, batch_size=self.batch_size)

    def test_unseen_dataloader(self):
        if self.test_unseen_dataset is None:
            raise ValueError("test unseen dataset is not loaded")
        return DataLoader(self.test_unseen_dataset, batch_size=self.batch_size)