import inspect
import re
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from captum.attr import LayerIntegratedGradients, LayerGradientXActivation
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer
from transformers_interpret.errors import (
    AttributionsNotCalculatedError,
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)
import numpy as np


class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: nn.Module, tokens: list):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.tokens = tokens

    def summarize(self, end_idx=None, flip_sign: bool = False):
        if flip_sign:
            multiplier = -1
        else:
            multiplier = 1
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0) * multiplier

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(
                zip([t for token in self.tokens for t in token], self.attributions_sum)
            ):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")


class LGXAAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        sep_id: int,
        attention_mask: torch.Tensor,
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ref_token_type_ids: Optional[torch.Tensor] = None,
        ref_position_ids: Optional[torch.Tensor] = None,
        internal_batch_size: Optional[int] = None,
        n_steps: int = 50,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.target = target
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.lgxa = LayerGradientXActivation(self.custom_forward, self.embeddings)

        self._attributions = self.lgxa.attribute(
            inputs=(self.input_ids),
            target=self.target,
            additional_forward_args=(self.attention_mask),
        )


class LIGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        sep_id: int,
        attention_mask: torch.Tensor,
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ref_token_type_ids: Optional[torch.Tensor] = None,
        ref_position_ids: Optional[torch.Tensor] = None,
        internal_batch_size: Optional[int] = None,
        n_steps: int = 50,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.target = target
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)

        if self.token_type_ids is not None and self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                    self.ref_position_ids,
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_position_ids,
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif self.token_type_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

        else:
            self._attributions, self.delta = self.lig.attribute(
                inputs=self.input_ids,
                baselines=self.ref_input_ids,
                target=self.target,
                return_convergence_delta=True,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(
                zip([t for token in self.tokens for t in token], self.attributions_sum)
            ):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None, flip_sign: bool = False):
        if flip_sign:
            multiplier = -1
        else:
            multiplier = 1
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0) * multiplier
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])
        self.attributions_sum = torch.flatten(self.attributions_sum)

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):
        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            self.delta,
        )


SUPPORTED_ATTRIBUTION_TYPES = ["lig", "lgxa", "lrp"]


class BaseExplainer(ABC):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if self.model.config.model_type == "gpt2":
            self.ref_token_id = self.tokenizer.eos_token_id
        else:
            self.ref_token_id = self.tokenizer.pad_token_id

        self.sep_token_id = (
            self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
        )
        self.cls_token_id = (
            self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        )

        self.model_prefix = model.base_model_prefix

        nonstandard_model_types = ["roberta"]
        if (
            self._model_forward_signature_accepts_parameter("position_ids")
            and self.model.config.model_type not in nonstandard_model_types
        ):
            self.accepts_position_ids = True
        else:
            self.accepts_position_ids = False

        if (
            self._model_forward_signature_accepts_parameter("token_type_ids")
            and self.model.config.model_type not in nonstandard_model_types
        ):
            self.accepts_token_type_ids = True
        else:
            self.accepts_token_type_ids = False

        self.device = self.model.device

        self.word_embeddings = self.model.get_input_embeddings()
        self.position_embeddings = None
        self.token_type_embeddings = None

        self._set_available_embedding_types()

    @abstractmethod
    def encode(self, text: str = None):
        """
        Encode given text with a model's tokenizer.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, input_ids: torch.Tensor) -> List[str]:
        """
        Decode received input_ids into a list of word tokens.


        Args:
            input_ids (torch.Tensor): Input ids representing
            word tokens for a sentence/document.

        """
        raise NotImplementedError

    @property 
    @abstractmethod
    def word_attributions(self):
        raise NotImplementedError

    @abstractmethod
    def _run(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def _forward(self):
        """
        Forward defines a function for passing inputs
        through a models's forward method.

        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_attributions(self):
        """
        Internal method for calculating the attribution
        values for the input text.

        """
        raise NotImplementedError

    def _make_input_reference_pair(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tokenizes `text` to numerical token id  representation `input_ids`,
        as well as creating another reference tensor `ref_input_ids` of the same length
        that will be used as baseline for attributions. Additionally
        the length of text without special tokens appended is prepended is also
        returned.

        Args:
            text (str): Text for which we are creating both input ids
            and their corresponding reference ids

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]
        """

        # if no special tokens were added
        ref_input_ids = input_ids.clone()
        ref_input_ids[(ref_input_ids != self.cls_token_id) & (ref_input_ids != self.sep_token_id)] = self.ref_token_id
        text_len = ((ref_input_ids != self.cls_token_id) & (ref_input_ids != self.sep_token_id)).sum().item()

        return (
            input_ids.to(self.device),
            ref_input_ids.to(self.device),
            text_len,
        )

    def _make_input_reference_token_type_pair(
        self, input_ids: torch.Tensor, sep_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two tensors indicating the corresponding token types for the `input_ids`
        and a corresponding all zero reference token type tensor.
        Args:
            input_ids (torch.Tensor): Tensor of text converted to `input_ids`
            sep_idx (int, optional):  Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([0 if i <= sep_idx else 1 for i in range(seq_len)], device=self.device).expand_as(
            input_ids
        )
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.device).expand_as(input_ids)

        return (token_type_ids, ref_token_type_ids)

    def _make_input_reference_position_id_pair(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tensors for positional encoding of tokens for input_ids and zeroed tensor for reference ids.

        Args:
            input_ids (torch.Tensor): inputs to create positional encoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        ref_position_ids = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return (position_ids, ref_position_ids)

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(input_ids)

    def _get_preds(
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        if self.accepts_position_ids and self.accepts_token_type_ids:
            preds = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            return preds

        elif self.accepts_position_ids:
            preds = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            return preds
        elif self.accepts_token_type_ids:
            preds = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

            return preds
        else:
            preds = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            return preds

    def _clean_text(self, text: str) -> str:
        text = re.sub("([.,!?()])", r" \1 ", text)
        text = re.sub("\s{2,}", " ", text)
        return text

    def _model_forward_signature_accepts_parameter(self, parameter: str) -> bool:
        signature = inspect.signature(self.model.forward)
        parameters = signature.parameters
        return parameter in parameters

    def _set_available_embedding_types(self):
        model_base = getattr(self.model, self.model_prefix)
        if self.model.config.model_type == "gpt2" and hasattr(model_base, "wpe"):
            self.position_embeddings = model_base.wpe.weight
        else:
            if hasattr(model_base, "embeddings"):
                self.model_embeddings = getattr(model_base, "embeddings")
                if hasattr(self.model_embeddings, "position_embeddings"):
                    self.position_embeddings = self.model_embeddings.position_embeddings
                if hasattr(self.model_embeddings, "token_type_embeddings"):
                    self.token_type_embeddings = self.model_embeddings.token_type_embeddings

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__}"
        s += ")"

        return s


class SequenceClassificationExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type
    `{MODEL_NAME}ForSequenceClassification` from the Transformers package.

    Calculates attribution for `text` using the given model
    and tokenizer.

    Attributions can be forced along the axis of a particular output index or class name.
    To do this provide either a valid `index` for the class label's output or if the outputs
    have provided labels you can pass a `class_name`.

    This explainer also allows for attributions with respect to a particlar embedding type.
    This can be selected by passing a `embedding_type`. The default value is `0` which
    is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
    If a model does not take position ids in its forward method (distilbert) a warning will
    occur and the default word_embeddings will be chosen instead.

    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
        custom_labels: Optional[List[str]] = None,
    ):
        """
        Args:
            model (PreTrainedModel): Pretrained huggingface Sequence Classification model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".
            custom_labels (List[str], optional): Applies custom labels to label2id and id2label configs.
                                                 Labels must be same length as the base model configs' labels.
                                                 Labels and ids are applied index-wise. Defaults to None.

        Raises:
            AttributionTypeNotSupportedError:
        """
        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        if custom_labels is not None:
            if len(custom_labels) != len(model.config.label2id):
                raise ValueError(
                    f"""`custom_labels` size '{len(custom_labels)}' should match pretrained model's label2id size
                    '{len(model.config.label2id)}'"""
                )

            self.id2label, self.label2id = self._get_id2label_and_label2id_dict(custom_labels)
        else:
            self.label2id = model.config.label2id
            self.id2label = model.config.id2label

        self.attributions: Union[None, LIGAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self._single_node_output = False

        self.internal_batch_size = None
        self.n_steps = 50

    @staticmethod
    def _get_id2label_and_label2id_dict(
        labels: List[str],
    ) -> Tuple[Dict[int, str], Dict[str, int]]:
        id2label: Dict[int, str] = dict()
        label2id: Dict[str, int] = dict()
        for idx, label in enumerate(labels):
            id2label[idx] = label
            label2id[label] = idx

        return id2label, label2id

    def encode(self, text: str = None) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        return self.tokenizer.convert_ids_to_tokens(input_ids)

    @property
    def predicted_class_index(self) -> int:
        "Returns predicted class index (int) for model with last calculated `input_ids`"
        if len(self.input_ids) > 0:
            # we call this before _forward() so it has to be calculated twice
            preds = self.model(self.input_ids)[0]
            self.pred_class = torch.argmax(torch.softmax(preds, dim=0)[0])
            return torch.argmax(torch.softmax(preds, dim=1)[0]).cpu().detach().numpy()

        else:
            raise InputIdsNotCalculatedError("input_ids have not been created yet.`")

    @property
    def predicted_class_name(self):
        "Returns predicted class name (str) for model with last calculated `input_ids`"
        try:
            index = self.predicted_class_index
            return self.id2label[int(index)]
        except Exception:
            return self.predicted_class_index

    @property
    def word_attributions(self) -> list:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."
        if self.attributions is not None:
            return self.attributions.word_attributions
        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")

    def visualize(self, html_filepath: str | None = None, true_class: str | None = None, **kwargs):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.

        If the true class is known for the text that can be passed to `true_class`

        """
        tokens = [token.replace("Ġ", "") for inputs in self.input_ids for token in self.decode(inputs)]
        attr_class = self.id2label[self.selected_index]

        if "predicted_class" in kwargs:
            predicted_class = kwargs["predicted_class"]
        elif self._single_node_output:
            if true_class is None:
                true_class = round(float(self.pred_probs))
            predicted_class = round(float(self.pred_probs))
            attr_class = round(float(self.pred_probs))
        else:
            if true_class is None:
                true_class = self.selected_index
            predicted_class = self.predicted_class_name

        score_viz = self.attributions.visualize_attributions(  # type: ignore
            self.pred_probs,
            predicted_class,
            true_class,
            attr_class,
            tokens,
        )
        html = viz.visualize_text([score_viz])

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        preds = self._get_preds(input_ids, token_type_ids, position_ids, attention_mask)
        preds = preds[0]

        # if it is a single output node
        if len(preds[0]) == 1:
            self._single_node_output = True
            self.pred_probs = torch.sigmoid(preds)[0][0]
            return torch.sigmoid(preds)[:, :]

        self.pred_probs = torch.softmax(preds, dim=1)[0][self.selected_index]
        return torch.softmax(preds, dim=1)[:, self.selected_index]

    def _calculate_attributions(self, embeddings: Embedding, index: int | None = None, class_name: str | None = None):  # type: ignore
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.input_ids)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        (
            self.token_type_ids,
            self.ref_token_type_ids,
        ) = self._make_input_reference_token_type_pair(self.input_ids, self.sep_idx)

        if index is not None:
            self.selected_index = index
        elif class_name is not None:
            if class_name in self.label2id.keys():
                self.selected_index = int(self.label2id[class_name])
            else:
                s = f"'{class_name}' is not found in self.label2id keys."
                s += "Defaulting to predicted index instead."
                warnings.warn(s)
                self.selected_index = int(self.predicted_class_index)
        else:
            self.selected_index = int(self.predicted_class_index)

        reference_tokens = [[token.replace("Ġ", "") for token in self.decode(input)] for input in self.input_ids]
        if self.attribution_type == "lig":
            lig = LIGAttributions(
                custom_forward=self._forward,
                embeddings=embeddings,
                tokens=reference_tokens,
                input_ids=self.input_ids,
                ref_input_ids=self.ref_input_ids,
                sep_id=self.sep_idx,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                ref_position_ids=self.ref_position_ids,
                token_type_ids=self.token_type_ids,
                ref_token_type_ids=self.ref_token_type_ids,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
            lig.summarize()
            self.attributions = lig
        elif self.attribution_type == "lgxa":
            lgxa = LGXAAttributions(
                custom_forward=self._forward,
                embeddings=embeddings,
                tokens=reference_tokens,
                input_ids=self.input_ids,
                ref_input_ids=self.ref_input_ids,
                sep_id=self.sep_idx,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                ref_position_ids=self.ref_position_ids,
                token_type_ids=self.token_type_ids,
                ref_token_type_ids=self.ref_token_type_ids,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
            lgxa.summarize()
            self.attributions = lgxa
        else:
            raise NotImplementedError

    def _run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        index: int | None = None,
        class_name: str | None = None,
        embedding_type: int | None = None,
    ) -> list:  # type: ignore
        if embedding_type is None:
            embeddings = self.word_embeddings
        else:
            if embedding_type == 0:
                embeddings = self.word_embeddings
            elif embedding_type == 1:
                if self.accepts_position_ids and self.position_embeddings is not None:
                    embeddings = self.position_embeddings
                else:
                    warnings.warn(
                        "This model doesn't support position embeddings for attributions. Defaulting to word embeddings"
                    )
                    embeddings = self.word_embeddings
            else:
                embeddings = self.word_embeddings

        self.input_ids = input_ids.to(self.device)
        self.attention_mask = attention_mask.to(self.device)

        self._calculate_attributions(embeddings=embeddings, index=index, class_name=class_name)
        return self.word_attributions  # type: ignore

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        index: int = None,
        class_name: str = None,
        embedding_type: int = 0,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> list:
        """
        Calculates attribution for `text` using the model
        and tokenizer given in the constructor.

        Attributions can be forced along the axis of a particular output index or class name.
        To do this provide either a valid `index` for the class label's output or if the outputs
        have provided labels you can pass a `class_name`.

        This explainer also allows for attributions with respect to a particular embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        Args:
            text (str): Text to provide attributions for.
            index (int, optional): Optional output index to provide attributions for. Defaults to None.
            class_name (str, optional): Optional output class name to provide attributions for. Defaults to None.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.
        Returns:
            list: List of tuples containing words and their associated attribution scores.
        """

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        return self._run(input_ids, attention_mask, index, class_name, embedding_type=embedding_type)

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s


def compute_lrp_explanation(
    model_components,
    model_inputs,
    logit_function=lambda x: x,
    proba_function=lambda x: x,
    return_gradient_norm=False,
):
    embeddings = model_components["embeddings"](
        input_ids=model_inputs["input_ids"], token_type_ids=model_inputs["token_type_ids"]
    )
    embeddings_ = embeddings.detach().requires_grad_(True)
    sequence_output = model_components["encoder"](embeddings_, model_inputs["attention_mask"])[0]
    pooled_output = model_components["agg"](model_components["pooler"](sequence_output))
    logits = model_components["classifier"](pooled_output)

    # Select what signal to propagate back through the network
    selected_logit = logit_function(logits)
    selected_logit.sum().backward()

    gradient = embeddings_.grad

    logits = logits.squeeze().detach().cpu()
    if return_gradient_norm:
        # Compute L1 norm over hidden dim
        gradient = gradient.squeeze().detach().cpu().numpy()
        relevance = np.linalg.norm(gradient, 1, -1)
        return relevance, logits.numpy()
    else:
        # Compute lrp explanation
        relevance = gradient * embeddings_
        # sum over embedding dimension
        relevance = relevance.sum(-1).squeeze().detach().cpu().numpy()
        return relevance, logits.numpy()


def get_lrp_ranks(model, tokenizer, sample, output, model_inputs, logit_function, **kwargs):
    model_components = {
        "embeddings": model.bert.embeddings,
        "encoder": model.bert.encoder,
        "pooler": model.bert.pooler,
        "classifier": model.classifier,
    }
    if not hasattr(model.config, 'aggregation'):
        model_components['agg'] = lambda x: x
    elif model.config.aggregation == "norm":
        model_components["agg"] = lambda x: model.norm(x.sum(0))
    elif model.config.aggregation == "avgmaxnorm_pool":
        model_components["agg"] = lambda x: model.norm(torch.cat([x.mean(0), x.max(0)[0]]))
    else:
        raise NotImplementedError

    relevance, logits = compute_lrp_explanation(
        model_components,
        model_inputs,
        logit_function=logit_function,
    )
    return relevance.flatten(), logits
