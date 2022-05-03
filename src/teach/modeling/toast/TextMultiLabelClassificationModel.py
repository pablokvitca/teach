import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import pytorch_lightning as pl
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
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

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.mlb = MultiLabelBinarizer(classes=range(self.num_labels))

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx, optimizer_idx=None) -> STEP_OUTPUT:
        res = self._single_step(batch, batch_idx)
        self.log("train_loss", res["loss"], batch_size=res["labels"].size(0))
        return res

    def training_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]):
        self._epoch_end("training", outputs)

    def _epoch_end(self, _type: str, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds = self.mlb.fit_transform(
            [[list(preds.cpu().numpy()) for preds in x["preds"]] for x in outputs][0]
        )
        labels = self.mlb.fit_transform(
            [[[label for label in labels if label != -1]
              for labels in x["labels"].long().tolist()]
             for x in outputs
            ][0])

        accuracy = accuracy_score(labels, preds)
        sampled_precision = precision_score(labels, preds, average="samples")
        sampled_recall = recall_score(labels, preds, average="samples")
        sampled_f1 = f1_score(labels, preds, average="samples")
        per_class_precision = precision_score(labels, preds, average=None)
        per_class_recall = recall_score(labels, preds, average=None)
        per_class_f1 = f1_score(labels, preds, average=None)

        self.log(f"{_type}/accuracy", accuracy, prog_bar=True)
        self.log(f"{_type}/sampled_precision", sampled_precision, prog_bar=True)
        self.log(f"{_type}/sampled_recall", sampled_recall, prog_bar=True)
        self.log(f"{_type}/sampled_f1", sampled_f1, prog_bar=True)
        for i, (class_precision, class_recall, class_f1) in enumerate(
                zip(per_class_precision, per_class_recall, per_class_f1)
        ):
            self.log(f"{_type}/precision/{i}", class_precision, prog_bar=True)
            self.log(f"{_type}/recall/{i}", class_recall, prog_bar=True)
            self.log(f"{_type}/f1/{i}", class_f1, prog_bar=True)

        self.log(f"{_type}/loss", loss, prog_bar=True)

        if _type == 'validation':
            self.log('val_loss', loss, prog_bar=True)

    def _single_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y = batch["labels"]
        outputs = self(batch)
        _, logits = outputs[0], outputs[1]

        loss = self.loss_fn(
            logits.view(-1, self.num_labels),
            y.float().view(-1, self.num_labels)) \
            if len(y) is not None else 0

        batch_size = y.size(0)
        preds = [
            torch.nonzero(logits[i].sigmoid() > self.label_threshold).squeeze()
            for i in range(batch_size)
        ]

        return {
            "loss": loss,
            "preds": preds,
            "labels": y
        }

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._single_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]):
        return self._epoch_end("validation", outputs)

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._single_step(batch, batch_idx)

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
