# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
from typing import List

import numpy as np
import torch

from teach.inference.actions import all_agent_actions, obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger
from teach.modeling.et.alfred.nn.transforms import Transforms
from teach.modeling.toast.NaiveMultimodalModel import NaiveMultiModalModel
from teach.modeling.toast.utils import get_text_tokens_from_instance, encode_as_word_vectors, pad_list

from gensim.models import KeyedVectors

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
        # logger.info("Initializing AgentModel...")
        # logger.info("\tParsing Arguments...")
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1, help="Random seed")
        parser.add_argument("--w2v_path", type=str, required=True, help="path to folder w2v .gz")
        parser.add_argument("--model_path", type=str, required=True, help="path to folder trained model")
        args = parser.parse_args(model_args)

        # logger.info(f"\tFirstModel using seed {args.seed}")
        np.random.seed(args.seed)

        # logger.info("\tLoading image transform...")
        self._img_transform = Transforms.get_transform("default")

        self.text_pad_size = 100

        self.prev_actions_pad_size = 100
        self.total_actions = len(all_agent_actions)

        # logger.info("\tLoading Word2Vec transform...")
        self.w2v_model = KeyedVectors.load_word2vec_format(args.w2v_path, binary=True, limit=100000)

        # logger.info("\tInitializing Naive Model...")
        self.model = NaiveMultiModalModel.load_from_checkpoint(args.model_path)
        self.model.eval()

        self.instance_text_encoded = None
        self.observed_actions = 0
        self.prev_actions = None

        # logger.info("Done!")

    def get_next_action(self, img, edh_instance, prev_action, img_name=None, edh_name=None):
        """
        TODO
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
        # logger.info("Selecting next action...")
        # logger.info("\tTensorizing image...")
        logger.info(f"\tINPUT IMAGE {img}")
        img_tensor = self.tensorize_image(img)
        # logger.info("\tComputing actions scores with model...")
        logger.info(f"\tINPUT IMAGE SHAPE {img_tensor.size()}")
        # logger.info(f"\tINPUT TEXT SHAPE {self.instance_text_encoded.size()}")
        # logger.info(f"\tINPUT PREV ACTIONS SHAPE {self.prev_actions.size()}")
        img_tensor = img_tensor[None, ...].float()
        action_probs = self.model.forward(img_tensor, self.instance_text_encoded, self.prev_actions)
        # logger.info(f"\tOUTPUT SHAPE {action_probs.size()}")
        action, one_hot_action = FirstModel._get_action_from_probs(action_probs)
        obj_relative_coord = None
        if action in obj_interaction_actions:
            obj_relative_coord = [
                np.random.uniform(high=0.99),
                np.random.uniform(high=0.99),
            ]
        self._add_to_prev_action(one_hot_action)
        # logger.info(f"SELECTED: {action}")
        return action, obj_relative_coord

    def tensorize_image(self, img):
        return self._img_transform(img)

    @staticmethod
    def _get_action_from_probs(probs):
        best_index = torch.argmax(probs)
        return all_agent_actions[best_index], best_index

    def _add_to_prev_action(self, one_hot_action_index):
        place_at = min(self.observed_actions, self.prev_actions_pad_size - 1)
        if self.observed_actions > self.prev_actions_pad_size:
            self.prev_actions.roll(-1, dims=1)
        action_one_hot = torch.zeros(self.total_actions)
        action_one_hot[one_hot_action_index] = 1
        self.prev_actions[:, place_at] = action_one_hot
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
        try:
            # logger.info("Starting EDH instance...")
            # logger.info("\tPreparing prev actions...")
            self.observed_actions = 0
            self.prev_actions = self._prev_actions_tensor_padded()
            # logger.info("\tPreparing text vectors...")
            text_from_instance = get_text_tokens_from_instance(edh_instance)
            text_from_instance = pad_list(text_from_instance, self.text_pad_size)
            self.instance_text_encoded = encode_as_word_vectors(self.w2v_model, text_from_instance)
            # logger.info("Done!")
            return True
        except Exception as err:
            logger.info(f"Could not start EDH instance: {err}")
            return False
