# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
from typing import List

import numpy as np
import torch

from teach.inference.actions import all_agent_actions, obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger
from teach.modeling.toast.NaiveMultimodalModel import NaiveMultiModalModel
from teach.modeling.toast.utils import get_text_tokens_from_instance, encode_as_word_vectors, pad_list

logger = create_logger(__name__)


class FirstModel(TeachModel):
    """
    Sample implementation of TeachModel.
    Demonstrates usage of custom arguments as well as sample implementation of get_next_actions method
    """

    def __init__(self, process_index: int, num_processes: int, model_args: List[str]):
        """Constructor

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1, help="Random seed")
        args = parser.parse_args(model_args)

        logger.info(f"FirstModel using seed {args.seed}")
        np.random.seed(args.seed)

        self.text_pad_size = 50

        self.prev_actions_pad_size = 50
        self.total_actions = len(all_agent_actions)

        self.model = NaiveMultiModalModel()  # TODO: params!
        self.instance_text_encoded = None
        self.observed_actions = 0
        self.prev_actions = None

    def get_next_action(self, img, edh_instance, prev_action, img_name=None, edh_name=None):
        """
        This method will be called at each timestep during inference to get the next predicted action from the model.
        :param img: PIL Image containing agent's egocentric image
        :param edh_instance: EDH instance
        :param prev_action: One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values
        from a previous call of get_next_action
        :param img_name: image file name
        :param edh_name: EDH instance file name
        :return action: An action name from all_agent_actions
        :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
        The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
        an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
        performed on it, and executes the action in AI2-THOR.
        """
        img_tensor = self.tensorize_image(img)
        action_probs = self.model.forward(img_tensor, self.instance_text_encoded, self.prev_actions)
        action, one_hot_action = FirstModel._get_action_from_probs(action_probs)
        obj_relative_coord = None
        if action in obj_interaction_actions:
            obj_relative_coord = [
                np.random.uniform(high=0.99),
                np.random.uniform(high=0.99),
            ]
        self._add_to_prev_action(one_hot_action)
        return action, obj_relative_coord

    def tensorize_image(self, img):
        pass  # TODO

    @staticmethod
    def _get_action_from_probs(probs):
        best_index = torch.argmax(probs)
        return all_agent_actions[best_index], best_index

    def _add_to_prev_action(self, one_hot_action_index):
        action_one_hot = torch.zeros(self.total_actions)
        action_one_hot[one_hot_action_index] = 1
        self.prev_actions[:, self.observed_actions] = action_one_hot
        self.observed_actions += 1

    def _prev_actions_tensor_padded(self):
        return torch.zeros((self.total_actions, self.prev_actions_pad_size))

    def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
        """
        Since this class produces random actions at every time step, no particular setup is needed. When running model
        inference, this would be a suitable place to preprocess the dialog, action and image history
        :param edh_instance: EDH instance
        :param edh_history_images: List of images as PIL Image objects (loaded from files in
                                   edh_instance['driver_image_history'])
        :param edh_name: EDH instance file name
        """
        self.observed_actions = 0
        self.prev_actions = self._prev_actions_tensor_padded()
        text_from_instance = get_text_tokens_from_instance(edh_instance)
        text_from_instance = pad_list(text_from_instance, self.text_pad_size)
        self.instance_text_encoded = encode_as_word_vectors(text_from_instance)
        return True
