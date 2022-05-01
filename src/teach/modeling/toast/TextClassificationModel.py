import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from typing import Union, List

from teach.logger import create_logger

logger = create_logger(__name__)


class TextClassificationModel(pl.LightningModule):

    def __init__(
            self,
            pretrained_model_name="distilbert-base-uncased",
            num_labels=5,
            learning_rate=0.001,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size=16,
            eval_batch_size=16,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels

        self.hg_config = AutoConfig.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, config=self.hg_config)

        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon,
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay,
        self.train_batch_size = train_batch_size,
        self.eval_batch_size = eval_batch_size,
        self.total_steps = None

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx, optimizer_idx=None) -> STEP_OUTPUT:
        # NOTE: if modified, also modify the validation step
        y = batch["labels"]
        outputs = self.forward(batch)
        loss = outputs[0]

        self.log("train_loss", loss, batch_size=y.size(0))

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # similar to the training step but stop on output of EOS token
        y = batch["labels"]
        outputs = self.forward(batch)
        loss, logits = outputs[0], outputs[1]
        preds = logits.argmax(dim=1)

        return {
            "loss": loss,
            "preds": preds,
            "labels": y
        }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds = torch.stack([x["preds"] for x in outputs])
        labels = torch.stack([x["labels"] for x in outputs])

        acc = ((preds.flatten() == labels.flatten()).sum().item()) / labels.flatten().size(0)

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

    def setup(self, stage=None):
        if stage == 'fit':
            train_loader = self.trainer.datamodule.train_dataloader()
            tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
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
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }