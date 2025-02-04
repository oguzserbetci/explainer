"""
This code has been adapted from https://github.com/coastalcph/humans-contrastive-xai
"""


from src import layer_utils
import torch
from typing import Optional, Tuple
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertPooler
from transformers import BertForSequenceClassification
from src.models import BertForLongerSequenceClassification

import math

class BertPoolerXAI(BertPooler):
    def __init__(self, config):
        super().__init__(config)
        self.activation = layer_utils.ActivationXAI(torch.tanh)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertSelfAttentionXAI(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Detach for LRP conservation
        attention_probs = attention_probs.detach()

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class LNargsDetach(object):
    def __init__(self):
        self.lnv = "bertnorm"
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.0
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True
        self.elementwise_affine = True


largs = LNargsDetach()


class BertForSequenceClassificationXAI(BertForSequenceClassification):
    def __init__(self, config, lrp=True):
        super().__init__(config)

        if lrp:
            self.override_default_layers(config)

    def override_default_layers(self, config):
        self.bert.pooler = BertPoolerXAI(config)
        for i in range(config.num_hidden_layers):
            self.bert.encoder.layer[i].attention.self = BertSelfAttentionXAI(config)

            self.bert.encoder.layer[i].attention.output.LayerNorm = layer_utils.LayerNormXAI(
                (config.hidden_size,), eps=config.layer_norm_eps, args=largs
            )

            self.bert.encoder.layer[i].intermediate.intermediate_act_fn = (
                layer_utils.GELUActivationXAI()
            )

            self.bert.encoder.layer[i].output.LayerNorm = layer_utils.LayerNormXAI(
                (config.hidden_size,), eps=config.layer_norm_eps, args=largs
            )


class BertForLongerSequenceClassificationXAI(BertForLongerSequenceClassification):
    def __init__(self, config, lrp=True):
        super().__init__(config)

        if lrp:
            self.override_default_layers(config)
        if config.aggregation == "norm":
            self.norm = layer_utils.LayerNormXAI(self.embedding_size, args=largs)
        elif config.aggregation == 'avgmaxnorm_pool':
            self.norm = layer_utils.LayerNormXAI(self.embedding_size, args=largs)

    def override_default_layers(self, config):
        self.bert.pooler = BertPoolerXAI(config)
        for i in range(config.num_hidden_layers):
            self.bert.encoder.layer[i].attention.self = BertSelfAttentionXAI(config)

            self.bert.encoder.layer[i].attention.output.LayerNorm = layer_utils.LayerNormXAI(
                (config.hidden_size,), eps=config.layer_norm_eps, args=largs
            )

            self.bert.encoder.layer[i].intermediate.intermediate_act_fn = (
                layer_utils.GELUActivationXAI()
            )

            self.bert.encoder.layer[i].output.LayerNorm = layer_utils.LayerNormXAI(
                (config.hidden_size,), eps=config.layer_norm_eps, args=largs
            )
    