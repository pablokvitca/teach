import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import pytorch_lightning as pl
from torch.optim import Adam
from transformers import AutoConfig, AutoModelForSequenceClassification
from typing import Union, List

from teach.logger import create_logger

logger = create_logger(__name__)


class TextClassificationModel(pl.LightningModule):

    def __init__(
            self,
            pretrained_model_name,
            num_labels,
            learning_rate=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels

        self.hg_config = AutoConfig.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, config=self.hg_config)

        self.learning_rate = learning_rate

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

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
