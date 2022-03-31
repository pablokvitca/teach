from torch import nn, cat

from teach.logger import create_logger
logger = create_logger(__name__)


class NaiveMultiModalModel(nn.Module):

    def __init__(
            self,
            image_conv_kwargs,
            image_hidden_layer_sizes,
            text_word_vec_size,
            text_input_words,
            text_hidden_layer_sizes,
            prev_actions_input_size,
            prev_actions_hidden_layer_sizes,
            combination_hidden_layers_size,
            output_layer_size,
            activations="relu",
    ):
        super(NaiveMultiModalModel, self).__init__()

        activation = NaiveMultiModalModel._get_activation_layer(activations)

        image_conv_seq_layers = []
        for kwargs in image_conv_kwargs:
            image_conv_seq_layers.append(nn.Conv2d(**kwargs))
            image_conv_seq_layers.append(activation())
        self.image_conv_input = nn.Sequential(
            *image_conv_seq_layers
        )

        image_seq_layers = []
        for in_size, out_size in image_hidden_layer_sizes:
            image_seq_layers.append(nn.Linear(in_size, out_size))
            image_seq_layers.append(activation())
        self.image_ffn_input = nn.Sequential(
            *image_seq_layers
        )

        text_input_size = text_word_vec_size * text_input_words
        prev_size = text_input_size
        image_seq_layers = []
        for size in text_hidden_layer_sizes:
            image_seq_layers.append(nn.Linear(prev_size, size))
            image_seq_layers.append(activation())
            prev_size = size
        self.text_input = nn.Sequential(
            *image_seq_layers
        )

        prev_action_seq_layers = []
        prev_size = prev_actions_input_size
        for size in prev_actions_hidden_layer_sizes:
            prev_action_seq_layers.append(nn.Linear(prev_size, size))
            prev_action_seq_layers.append(activation())
            prev_size = size
        self.prev_actions_input = nn.Sequential(
            *prev_action_seq_layers
        )

        comb_seq_layers = []
        prev_size = combination_hidden_layers_size[0]
        for size in combination_hidden_layers_size[1:]:
            comb_seq_layers.append(nn.Linear(prev_size, size))
            comb_seq_layers.append(activation())
            prev_size = size
        comb_seq_layers.append(nn.Linear(prev_size, output_layer_size))
        comb_seq_layers.append(nn.Softmax())  # Final activation
        self.comb = nn.Sequential(
            *comb_seq_layers
        )

    @staticmethod
    def _get_activation_layer(activation="relu"):
        return nn.ReLU  # TODO: activation layer

    def forward(self, current_image_tensor, text_tensor, previous_actions_tensor):
        image_conv_h = self.image_conv_input(current_image_tensor[None, ...].float())
        image_h = self.image_ffn_input(image_conv_h.flatten())
        text_h = self.text_input(text_tensor.flatten())
        prev_action_h = self.prev_actions_input(previous_actions_tensor.flatten())
        logger.info(f"IMAGE H SHAPE: {image_h.size()}")
        logger.info(f"TEXT H SHAPE: {text_h.size()}")
        logger.info(f"PREV ACTION H SHAPE: {prev_action_h.size()}")
        comb_h = cat((image_h, text_h, prev_action_h), 0)
        z_out = self.comb(comb_h)
        return z_out
