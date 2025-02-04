import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import datasets
import numpy as np
import torch
from datasets import DatasetDict

from sklearn.metrics import precision_recall_fscore_support
import transformers
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    BertForSequenceClassification,
    BertConfig,
)

from src.models import BertForLongerSequenceClassification, BertForLongerSequenceClassificationConfig
from src.configurator import BaseConfig, DataConfig
from src.data import DataCollatorForMultiSentenceMultitask
import src.data_processing as data_processing  # noqa: F401

os.environ["WANDB_PROJECT"] = "md"
IDENTIFIER = "_id"


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "GerMedBERT/medbert-512"
    aggregation: Literal["attention", "avg_pool", "norm_sum_pool", "norm", "lstm"] | None = None
    classifier_dropout: float = 0.2
    output_attentions: bool = False


@dataclass
class TransformersConfig(BaseConfig):
    model_config_class = ModelConfig
    data_config_class = DataConfig
    training_config_class = TrainingArguments


if __name__ == "__main__":

    def do_predict(dataset, prefix):
        results = trainer.predict(dataset, ["pred_loss", "adversary_loss", "loss", "hidden_states"])

        filtered_columns = list(config.data.tasks.keys())
        if IDENTIFIER in dataset.column_names:
            filtered_columns.append(IDENTIFIER)
        dataset = dataset.select_columns(filtered_columns)
        output_names = ["logits", "pooled_output", "deep_repr"]
        for output, name in zip(results.predictions, output_names):
            if name == "logits":
                columns = {}
                offset = 0
                for task_name, n_labels in config.data.tasks.items():
                    columns[f"{task_name}"] = [label for label in dataset[task_name]]
                    columns[f"{task_name}_pred"] = output[:, offset : offset + n_labels].argmax(-1)
                    offset += n_labels
            else:
                columns = {name: output}

            for name, column in columns.items():
                if not isinstance(column, list):
                    column = column.tolist()
                if name in dataset.column_names:
                    dataset = dataset.remove_columns(name)
                    dataset = dataset.add_column(name, column)
                else:
                    dataset = dataset.add_column(name, column)

        metrics = results.metrics
        output_dir = Path(config.training.output_dir)
        dataset.save_to_disk(str(output_dir / f"{prefix}_predict"))
        trainer.log_metrics(f"{prefix}_predict", metrics)
        trainer.save_metrics(f"{prefix}_predict", metrics)

    config = TransformersConfig.from_json(os.path.abspath(sys.argv[-1]))

    log_path = Path(
        f'{config.logging_dir}/train_multitask_{datetime.today().strftime("%Y-%m-%d")}.log'
    )
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger("transformers")
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)

    dataset = DatasetDict.load_from_disk(config.data.data_path)

    @torch.no_grad()
    def compute_metrics(eval_pred):
        all_predictions, all_labels = eval_pred
        if len(config.training.label_names) >1:
            overflow_to_sample_mapping = all_labels[1]
            overflow_label_mask = np.concatenate([[True], overflow_to_sample_mapping[1:] != overflow_to_sample_mapping[:-1]])
            labels = all_labels[0][overflow_label_mask]
        else:
            labels = all_labels

        if isinstance(all_predictions, tuple):
            all_predictions = all_predictions[0]
        
        if len(labels.shape) == 1:
            labels = labels[:, None]

        metrics = {}
        iter = [(all_predictions, labels, config.data.tasks, "model")]
        for predictions, labels, _tasks, metric_prefix in iter:
            current_task_offset = 0
            for i, (task, num_labels) in enumerate(_tasks.items()):
                task_predictions = predictions[
                    :, current_task_offset : current_task_offset + num_labels
                ].argmax(-1)
                task_labels = labels[:, i]
                scores = precision_recall_fscore_support(
                    task_labels, task_predictions, average=None, labels=range(num_labels)
                )
                metrics |= {
                    f"{metric_prefix}/{task}/{metric}/label={label}": score_for_label.item()
                    for metric, values in zip(["precision", "recall", "f1", "support"], scores)
                    for label, score_for_label in enumerate(values, 1)
                }
                scores = precision_recall_fscore_support(
                    task_labels, task_predictions, average="macro", labels=range(num_labels)
                )
                metrics |= {
                    f"{metric_prefix}/{task}/{metric}/macro": (
                        score if score is not None else None
                    )
                    for metric, score in zip(["precision", "recall", "f1"], scores)
                }

                scores = precision_recall_fscore_support(
                    task_labels, task_predictions, average="micro", labels=range(num_labels)
                )
                metrics |= {
                    f"{metric_prefix}/{task}/{metric}/micro": (
                        score if score is not None else None
                    )
                    for metric, score in zip(["precision", "recall", "f1"], scores)
                }
                scores = precision_recall_fscore_support(
                    task_labels, task_predictions, average="weighted", labels=range(num_labels)
                )
                metrics |= {
                    f"{metric_prefix}/{task}/{metric}/weighted": (
                        score if score is not None else None
                    )
                    for metric, score in zip(["precision", "recall", "f1"], scores)
                }

            for metric in ["precision", "recall", "f1"]:
                metrics[f"{metric_prefix}/{metric}/macro"] = np.mean(
                    [metrics[f"{metric_prefix}/{task}/{metric}/macro"] for task in _tasks.keys()]
                )
        if "adv_loss" in metrics:
            metrics["combined_loss"] = (
                metrics["model/loss"]
                + (metrics["model/loss"] / metrics["adv_loss"])
                - config.adversary_alpha * metrics["adv_loss"]
            )
        return metrics

    dataset = dataset.filter(lambda x: x['text'])

    if config.data.full_data:
        train_set = datasets.concatenate_datasets([dataset["train"], dataset["test"]])
    else:
        train_set = dataset["train"]
    eval_set = dataset[config.data.eval_split]

    if config.data.process_func:
        train_set = train_set.map(eval(config.data.process_func), fn_kwargs={"tokenizer": tokenizer})
        eval_set = eval_set.map(eval(config.data.process_func), fn_kwargs={"tokenizer": tokenizer})

    if config.data.dev:
        train_set = train_set.select(range(100))
        eval_set = eval_set.select(range(100))

    bert_config = BertForLongerSequenceClassificationConfig.from_pretrained(
    # bert_config = BertConfig.from_pretrained(
        **asdict(config.model),
        # tasks=config.data.tasks,
        num_labels=sum(config.data.tasks.values()),
    )
    bert_config.tasks = config.data.tasks

    # model = BertForSequenceClassification.from_pretrained(
    #     config.model.pretrained_model_name_or_path, config=bert_config
    # )

    model = BertForLongerSequenceClassification.from_pretrained(
        config.model.pretrained_model_name_or_path, config=bert_config
    )
    
    if config.model.aggregation is None:
        data_collator = transformers.DataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=model.config.max_position_embeddings,
            padding=True,
        )
    else:
        data_collator = DataCollatorForMultiSentenceMultitask(
            label_format=config.data.label_format,
            tasks=config.data.tasks,
            tokenizer=tokenizer,
        )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
    trainer = Trainer(
        model=model,
        args=config.training,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    if config.training.do_train:
        trainer.train()
        
    if config.training.do_predict:
        do_predict(
            eval_set,
            config.data.eval_split,
        )
