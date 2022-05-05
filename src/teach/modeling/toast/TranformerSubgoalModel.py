from itertools import chain

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Union, List

from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSeq2SeqLM

from teach.logger import create_logger
from teach.modeling.toast.Lang import Lang
from torchtext.data.metrics import bleu_score

logger = create_logger(__name__)


class TransformerSubgoalModel(pl.LightningModule):
    def __init__(
            self,
            input_lang: Lang,
            output_lang: Lang,
            encoder_hidden_size: int,
            decoder_hidden_size: int,
            tokenizer,
            teacher_forcing=False,
            learning_rate=0.001,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            model_name='BartForConditionalGeneration',
            num_beams=1,
            max_source_length=1024,
            max_target_length=512,
            ignore_pad_token_for_loss=True,
            preprocessing_num_workers=1,
            finetune_model=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_lang: Lang = input_lang
        self.output_lang: Lang = output_lang
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.tokenizer = tokenizer

        self.teacher_forcing = teacher_forcing
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.model_name = model_name
        self.num_beams = num_beams
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.preprocessing_num_workers = preprocessing_num_workers

        self.hg_config = AutoConfig.from_pretrained(self.pretrained_model_name)
        self.finetune_model = finetune_model
        if self.finetune_model:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name, config=self.hg_config)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hg_config)

        self.model.resize_token_embeddings(len(self.tokenizer))


    def forward(self, x):
        pass  # TODO: implement

    def training_step(self, batch, batch_idx, optimizer_idx=None) -> STEP_OUTPUT:
        x, y = batch[0]

        # TODO: implement

        self.log(f"training_step_loss", loss, prog_bar=True)

        return {
            "predicted": predicted,
            "reference": reference,
            "loss": loss
        }

    def _inf_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch[0]

        # TODO: implement

        predicted = [[self.output_lang.index2word[i] for i in output if i != self.output_lang.PAD_token_index] + [self.output_lang.EOS_token]]
        reference = [[[self.output_lang.index2word[y_token.item()] for y_token in y[i]]] for i in range(y.size(0))]

        return {
            "predicted": predicted,
            "reference": reference,
            "loss": loss
        }

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._inf_step(batch, batch_idx)

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._inf_step(batch, batch_idx)

    def _epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]], split: str) -> None:
        try:
            loss = torch.stack([output["loss"] for output in outputs]).mean()
            self.log(f"{split}/loss", loss)
            if split == "validation":
                self.log(f"val_loss", loss, prog_bar=True)

            if split != "train":
                predicted = list(chain(*[output["predicted"] for output in outputs]))
                reference = list(chain(*[output["reference"] for output in outputs]))
                bleu = bleu_score(predicted, reference)
                self.log(f"{split}/bleu_score", bleu, prog_bar=True)
        except:
            pass

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        return self._epoch_end(outputs, "validation")

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        return self._epoch_end(outputs, "validation_unseen")

    def setup(self, stage=None):
        if stage == 'fit':
            train_loader = self.trainer.datamodule.train_dataloader()
            tb_size = self.hparams.train_batch_size * max(1, self.trainer.num_devices)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        linear_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            "scheduler": linear_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]