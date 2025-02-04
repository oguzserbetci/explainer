import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch
from torch import nn
from transformers import (BertConfig, BertModel, BertPreTrainedModel,
                          LongformerModel, LongformerPreTrainedModel)
from transformers.modeling_outputs import SequenceClassifierOutput

# configure logging at the root level of Lightning
logger = logging.getLogger("lightning.pytorch")


class BertForLongerSequenceClassificationConfig(BertConfig):
    tasks: dict
    aggregation: Literal["attention", "pool", "norm", "lstm", "first"] = "norm"
    classifier_dropout: float = 0.2


@dataclass
class MultiTaskSequenceClassifierOutput(SequenceClassifierOutput):
    pooled_output: torch.FloatTensor = None
    deep_repr: Optional[torch.FloatTensor] = None


def normed_sum_pooling(input: torch.FloatTensor, eps=1e-5):
    """
    input: B x H
    Pham et al., “DeepCare: A Deep Dynamic Memory Model for Predictive Medicine.”
    """
    return input.sum(axis=0) / torch.clamp(torch.sqrt(input.sum(axis=0).abs()), min=eps)


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = None

    def forward(self, x):
        self.embedding = x
        return x




class BertForLongerSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, current_task=None):
        super().__init__(config)
        self.config = config

        self.num_labels = config.num_labels
        labels = [
            f"{task}={task_class}" for task, num_labels in config.tasks.items() for task_class in range(num_labels)
        ]
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.bert = BertModel(config)

        self.classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(self.classifier_dropout)

        self.recorder = Recorder()
        self.embedding_size = self.config.hidden_size
        if config.aggregation == "norm":
            self.norm = nn.LayerNorm(self.embedding_size)
        elif config.aggregation == "attention":
            self.embedding_size = 128 * 7
            self.query = nn.Linear(self.config.hidden_size, self.embedding_size, bias=False)
            self.key = nn.Linear(self.config.hidden_size, self.embedding_size, bias=False)
            self.value = nn.Linear(self.config.hidden_size, self.embedding_size, bias=False)
            self.attention = nn.MultiheadAttention(self.embedding_size, num_heads=7, dropout=config.classifier_dropout)
        elif config.aggregation == "norm_sum_pool":
            pass
        elif self.config.aggregation == "avgmax_pool":
            self.embedding_size *= 2
        elif self.config.aggregation == "avgmaxnorm_pool":
            self.embedding_size *= 2
            self.norm = nn.LayerNorm(self.embedding_size)
        elif config.aggregation == "avg_pool":
            pass
        elif config.aggregation is None:
            pass
        else:
            raise NotImplementedError

        self.classifier = nn.Sequential(
            OrderedDict(
                {
                    "0": nn.Dropout(config.classifier_dropout),
                    "1": nn.Linear(self.embedding_size, self.num_labels),
                }
            )
        )

        label_offset = 0
        if config.tasks is not None:
            self.task_labels = [
                f"{task}={label}"
                for task, num_labels in config.tasks.items()
                for label in range(0, num_labels + label_offset)
            ]
        self.current_task = current_task
        self.loss_fct = nn.CrossEntropyLoss(reduction="mean")

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        adversary_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        overflow_to_sample_mapping: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MultiTaskSequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        if (self.config.aggregation is not None) or (overflow_to_sample_mapping is not None):
            if overflow_to_sample_mapping is None:
                overflow_to_sample_mapping = torch.zeros(len(input_ids))
            sample_idx = torch.unique(overflow_to_sample_mapping, sorted=True)
            sentence_count = len(sample_idx)
            sequence_embeddings = torch.zeros(sentence_count, self.embedding_size, device=pooled_output.device)
            if labels is not None:
                _labels = torch.zeros(sentence_count, labels.shape[-1], device=pooled_output.device).type_as(labels)
            for i, overflow_to_sample in enumerate(sample_idx):
                # sequence_embeddings[i] = pooled_output[overflow_to_sample_mapping == i].sum(0).clamp(min=0)
                # sequence_embeddings[i] /= torch.sqrt(torch.clamp(sequence_embeddings[i], min=1e-9))
                sequence_embedding = pooled_output[overflow_to_sample_mapping == overflow_to_sample]
                if self.config.aggregation == "norm":
                    sequence_embedding = sequence_embedding.sum(axis=0)
                    sequence_embeddings[i] = self.norm(sequence_embedding)
                elif self.config.aggregation == "first":
                    sequence_embeddings[i] = sequence_embedding[0]
                elif self.config.aggregation == "attention":
                    q = self.query(sequence_embedding)
                    k = self.key(sequence_embedding)
                    v = self.value(sequence_embedding)
                    attn_o, attn_w = self.attention(q, k, v, average_attn_weights=False)
                    sequence_embeddings[i] = normed_sum_pooling(attn_o)
                elif self.config.aggregation == "norm_sum_pool":
                    sequence_embeddings[i] = normed_sum_pooling(sequence_embedding)
                elif self.config.aggregation == "avg_pool":
                    sequence_embeddings[i] = sequence_embedding.mean(0)
                elif self.config.aggregation == "avgmax_pool":
                    sequence_embeddings[i] = torch.cat([sequence_embedding.mean(0), sequence_embedding.max(0)[0]])
                elif self.config.aggregation == "avgmaxnorm_pool":
                    sequence_embedding = torch.cat([sequence_embedding.mean(0), sequence_embedding.max(0)[0]])
                    sequence_embeddings[i] = self.norm(sequence_embedding)
                else:
                    raise NotImplementedError
                if labels is not None:
                    _labels[i] = labels[overflow_to_sample_mapping == overflow_to_sample][0, :]
        else:
            sequence_embeddings = pooled_output
            if labels is not None:
                _labels = labels
        if labels is not None:
            labels = _labels
        logits = self.classifier(sequence_embeddings)

        pred_loss = None
        if labels is not None:
            if self.config.tasks is not None:
                pred_loss = loss_for_tasks(self.config.tasks, logits, labels, self.loss_fct)
            else:
                pred_loss = self.loss_fct(logits, labels)

        return MultiTaskSequenceClassifierOutput(
            loss=pred_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pooled_output=sequence_embeddings,
            deep_repr=self.recorder.embedding,
        )
    

def loss_for_tasks(tasks, logits, labels, loss_fct):
    loss = 0
    current_offset = 0
    for i, (task, num_labels) in enumerate(tasks.items()):
        _logits = logits[:, current_offset : current_offset + num_labels]
        current_offset += num_labels
        loss = loss + loss_fct(_logits, labels[:, i])
    return loss / len(tasks.keys())


def pred_for_tasks(tasks, logits):
    preds = []
    current_offset = 0
    for i, (task, num_labels) in enumerate(tasks.items()):
        _logits = logits[:, current_offset : current_offset + num_labels]
        current_offset += num_labels
        pred = _logits.argmax(-1).detach().cpu().squeeze().tolist()
        preds.append(pred)
    return preds


def logits_for_tasks(tasks, logits):
    task_logits = []
    current_offset = 0
    for i, (task, num_labels) in enumerate(tasks.items()):
        _logits = logits[:, current_offset : current_offset + num_labels]
        current_offset += num_labels
        task_logits.append(torch.softmax(_logits, dim=-1).tolist())
    return task_logits
