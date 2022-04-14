import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from teach.logger import create_logger
logger = create_logger(__name__)


class GRUTextOnlyModel(pl.LightningModule):

    def __init__(
            self,
            input_lang_n_words,
            encoder_hidden_size,
            decoder_hidden_size,
            output_lang_n_words,
            teacher_forcing=False,
            decoder_dropout_p=0.1,
            learning_rate=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_lang_n_words = input_lang_n_words
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_lang_n_words = output_lang_n_words
        self.teacher_forcing = teacher_forcing
        self.decoder_dropout_p = decoder_dropout_p
        self.learning_rate = learning_rate

        self.encoder = EncoderRNN(
            self.input_lang_n_words,
            self.encoder_hidden_size
        )

        self.decoder = AttnDecoderRNN(
            self.decoder_hidden_size,
            self.output_lang_n_words,
            dropout_p=self.decoder_dropout_p
        )

        self.SOS_token = 0  # TODO: init sos token!
        self.EOS_token = 1  # TODO: init eos token!

    def forward(
            self,
            current_image_tensor,
            text_tensor,
            previous_actions_tensor,
            pre_encoder_output=None,
            pre_decoder_output=None,
            return_only_action_probs=True
    ):

        if pre_encoder_output is None:
            encoder_outputs = torch.zeros(self.max_length, self.encoder_hidden_size, device=self.device)
            encoder_hidden = self.encoder.initHidden()
            for input_token_tensor_idx in range(text_tensor.size()[0]):
                encoder_output, encoder_hidden = self.encoder.forward(
                    text_tensor[input_token_tensor_idx],
                    encoder_hidden
                )
                encoder_outputs[input_token_tensor_idx] += encoder_output[0, 0]
        else:
            encoder_outputs, encoder_hidden = pre_encoder_output

        # Setup input for decoder
        if pre_decoder_output is None:
            decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
            decoder_hidden = encoder_hidden
        else:
            decoder_input, decoder_hidden = pre_decoder_output

        # Call decoder.forward - Only do for "next token"
        decoder_output, decoder_hidden, _ = self.decoder.forward(
            decoder_input,
            decoder_hidden,
            encoder_outputs,
        )

        if return_only_action_probs:
            return decoder_output
        else:
            return (encoder_outputs, encoder_hidden), (decoder_output, decoder_hidden)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        x_text = x["text"]
        pre_encoder_output, pre_decoder_output = None, None
        loss = torch.Tensor(0, device=self.device)
        for y_token_idx in range(y.size()[0]):
            y_token = y[y_token_idx]
            pre_encoder_output, (decoder_output, decoder_hidden) = \
                self.forward(
                    None,  # image input ignored
                    x_text,
                    None,  # prev action input ignored
                    pre_encoder_output=pre_encoder_output,
                    pre_decoder_output=pre_decoder_output,
                    return_only_action_probs=False
                )
            loss += F.cross_entropy(decoder_output, y_token)

            if self.teacher_forcing:
                decoder_input = y_token
            else:
                decoder_input = decoder_output.topk(1)[1].squeeze().detach()
                if decoder_input.item() == self.EOS_token:
                    break
            pre_decoder_output = (decoder_input, decoder_hidden)
        return loss if (not torch.isnan(loss) and not torch.isinf(loss)) else None

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.encoder.parameters(), lr=self.learning_rate),\
               Adam(self.decoder.parameters(), lr=self.learning_rate)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
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

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
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

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)