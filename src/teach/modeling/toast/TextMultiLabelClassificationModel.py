import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from typing import Union, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from teach.logger import create_logger

logger = create_logger(__name__)


class TextMultiLabelClassificationModel(pl.LightningModule):

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
            freeze_encoder=False,
            label_threshold=0.5,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels

        self.hg_config = AutoConfig.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, config=self.hg_config)

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon,
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay,
        self.train_batch_size = train_batch_size,
        self.eval_batch_size = eval_batch_size,
        self.total_steps = None

        self.label_threshold = label_threshold

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx, optimizer_idx=None) -> STEP_OUTPUT:
        # NOTE: if modified, also modify the validation step
        y = batch["labels"]
        outputs = self(batch)
        _, logits = outputs[0], outputs[1]

        loss = self.loss_fct(
            logits.view(-1, self.num_labels),
            y.float().view(-1, self.num_labels)) \
            if len(y) is not None else 0
        preds = torch.where(logits.sigmoid(dim=1) > self.label_threshold, 1, 0)

        self.log("train_loss", loss, batch_size=y.size(0))

        return {
            "loss": loss,
            "preds": preds,
            "labels": y
        }

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]):
        self._epoch_end("training", outputs)

    def _epoch_end(self, _type: str, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs]).flatten()
        labels = torch.cat([x["labels"] for x in outputs]).flatten()

        count_correct = (preds == labels).sum().item()
        total_counts = labels.size(0)
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")
        f1 = f1_score(labels, preds, average="macro")

        self.log(f"{_type}/accuracy", accuracy, prog_bar=True)
        self.log(f"{_type}/precision", precision, prog_bar=True)
        self.log(f"{_type}/recall", recall, prog_bar=True)
        self.log(f"{_type}/f1", f1, prog_bar=True)
        self.log(f"{_type}/count_correct", count_correct, prog_bar=True)
        self.log(f"{_type}/total_counts", total_counts, prog_bar=True)

        self.log(f"{_type}/loss", loss, prog_bar=True)

        if _type == 'validation':
            self.log('val_loss', loss, prog_bar=True)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # similar to the training step but stop on output of EOS token
        y = batch["labels"]
        outputs = self(batch)
        _, logits = outputs[0], outputs[1]

        loss = self.loss_fct(
            logits.view(-1, self.num_labels),
            y.float().view(-1, self.num_labels)) \
            if len(y) is not None else 0
        preds = torch.where(logits.sigmoid(dim=1) > self.label_threshold, 1, 0)

        return {
            "loss": loss,
            "preds": preds,
            "labels": y
        }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]):
        return self._epoch_end("validation", outputs)

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # similar to the training step but stop on output of EOS token
        y = batch["labels"]
        outputs = self(batch)
        loss, logits = outputs[0], outputs[1]
        preds = logits.argmax(dim=1)

        return {
            "loss": loss,
            "preds": preds,
            "labels": y
        }

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]):
        return self._epoch_end(f"validation_unseen", outputs)

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
