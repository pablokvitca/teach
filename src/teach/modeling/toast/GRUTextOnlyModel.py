import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from teach.logger import create_logger
from teach.modeling.toast.Lang import Lang

logger = create_logger(__name__)


class GRUTextOnlyModel(pl.LightningModule):

    def __init__(
            self,
            input_lang: Lang,
            output_lang: Lang,
            encoder_hidden_size: int,
            decoder_hidden_size: int,
            teacher_forcing=False,
            decoder_dropout_p=0.1,
            learning_rate=0.001,
            max_length=1000,
            max_output_length=1000,
            use_single_optimizer=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_lang: Lang = input_lang
        self.output_lang: Lang = output_lang
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.teacher_forcing = teacher_forcing
        self.decoder_dropout_p = decoder_dropout_p
        self.learning_rate = learning_rate
        self.max_length = max_length

        self.encoder = EncoderRNN(
            self.input_lang.n_words,
            self.encoder_hidden_size
        )

        self.decoder = AttnDecoderRNN(
            self.decoder_hidden_size,
            self.output_lang.n_words,
            dropout_p=self.decoder_dropout_p,
            max_length=self.max_length
        )

        self.use_single_optimizer = use_single_optimizer

    def forward(
            self,
            current_image_tensor,
            text_tensor,
            previous_actions_tensor,
            pre_encoder_output=None,
            pre_decoder_output=None,
            return_only_action_probs=True
    ):
        batch_size = text_tensor.size(1)
        if pre_encoder_output is None:
            encoder_outputs = torch.zeros(self.max_length, self.encoder_hidden_size, device=self.device)
            encoder_hidden = self.encoder.init_hidden(batch_size)
            for input_token_tensor_idx in range(text_tensor.size(0)):
                encoder_input = text_tensor[input_token_tensor_idx]
                if encoder_input.dim() == 0:
                    encoder_input = encoder_input.unsqueeze(0)
                encoder_output, encoder_hidden = self.encoder.forward(
                    encoder_input,
                    encoder_hidden
                )
                encoder_outputs[input_token_tensor_idx] += encoder_output[0, 0]
        else:
            encoder_outputs, encoder_hidden = pre_encoder_output

        # Setup input for decoder
        if pre_decoder_output is None:
            decoder_input = torch.tensor([[self.input_lang.SOS_token_index] for _ in range(batch_size)], device=self.device)
            decoder_hidden = encoder_hidden
        else:
            decoder_input, decoder_hidden = pre_decoder_output

        # Call decoder.forward - Only do for "next token"
        if decoder_input.dim() == 0:
            decoder_input = decoder_input.unsqueeze(0)
        decoder_output, decoder_hidden, _ = self.decoder.forward(
            decoder_input,
            decoder_hidden,
            encoder_outputs,
        )

        if return_only_action_probs:
            return decoder_output
        else:
            return (encoder_outputs, encoder_hidden), (decoder_output, decoder_hidden)

    def training_step(self, batch, batch_idx, optimizer_idx=None) -> STEP_OUTPUT:
        # NOTE: if modified, also modify the validation step
        x, y = batch[0]
        x_text = x
        pre_encoder_output, pre_decoder_output = None, None
        loss = torch.zeros(1, device=self.device)
        for y_token_idx in range(y.size(0)):
            y_token = y[y_token_idx]
            if y_token.dim() == 0:
                y_token = y_token.unsqueeze(0)
            pre_encoder_output, (decoder_output, decoder_hidden) = \
                self.forward(
                    None,  # image input ignored
                    x_text,
                    None,  # prev action input ignored
                    pre_encoder_output=pre_encoder_output,
                    pre_decoder_output=pre_decoder_output,
                    return_only_action_probs=False
                )

            loss += F.cross_entropy(
                decoder_output,
                F.one_hot(y_token, num_classes=self.output_lang.n_words).to(device=self.device, dtype=torch.float).squeeze(dim=1)
            )

            if self.teacher_forcing:
                decoder_input = y_token
            else:
                # TODO: fix batch > 1 will fail if not using teacher forcing
                decoder_input = decoder_output.topk(1)[1].squeeze().detach()
                if decoder_input.item() == self.output_lang.EOS_token_index:
                    break
            pre_decoder_output = (decoder_input, decoder_hidden)
        return loss.squeeze()

    def validation_step(self, batch, batch_idx):
        # similar to the training step but stop on output of EOS token
        x, y = batch[0]
        x_text = x
        pre_encoder_output, pre_decoder_output = None, None
        loss = torch.zeros(1, device=self.device)
        did_output_eos = False
        y_token_idx = 0
        while not did_output_eos and y_token_idx < self.max_output_length:
            y_token = y[y_token_idx] if y_token_idx < y.size(0) else \
                torch.Tensor([self.output_lang.EOS_token_index]).to(device=self.device, dtype=torch.long).unsqueeze(0)
            if y_token.dim() == 0:
                y_token = y_token.unsqueeze(0)
            y_token_idx += 1
            pre_encoder_output, (decoder_output, decoder_hidden) = \
                self.forward(
                    None,  # image input ignored
                    x_text,
                    None,  # prev action input ignored
                    pre_encoder_output=pre_encoder_output,
                    pre_decoder_output=pre_decoder_output,
                    return_only_action_probs=False
                )

            loss += F.cross_entropy(
                decoder_output,
                F.one_hot(y_token, num_classes=self.output_lang.n_words).to(device=self.device, dtype=torch.float).squeeze(dim=1)
            )

            decoder_input = decoder_output.topk(1)[1].squeeze().detach()

            if decoder_input.item() == self.output_lang.EOS_token_index:
                did_output_eos = True

            pre_decoder_output = (decoder_input, decoder_hidden)
        loss = loss.squeeze()

        # TODO: compute (and log, see below) accuracy, precision, recall, f1, hmm but sequence?
        # TODO: find sequential metric to report instead

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.use_single_optimizer:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            return Adam(self.encoder.parameters(), lr=self.learning_rate), \
                   Adam(self.decoder.parameters(), lr=self.learning_rate)


class EncoderRNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, _input: Tensor, hidden):
        batch_size = _input.size(0)
        embedded = self.embedding(_input).view(1, batch_size, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class AttnDecoderRNN(pl.LightningModule):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, _input, hidden, encoder_outputs):
        batch_size = _input.size(0)
        embedded = self.embedding(_input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
