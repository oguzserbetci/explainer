from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import ClassVar, Literal, Type
from transformers import PretrainedConfig

class BinderConfig(PretrainedConfig):
    '''copied from binder'''

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        cache_dir=None,
        revision="main",
        use_auth_token=False,
        hidden_dropout_prob=0.1,
        max_span_width=30,
        use_span_width_embedding=False,
        linear_size=128,
        init_temperature=0.07,
        start_loss_weight=0.2,
        end_loss_weight=0.2,
        span_loss_weight=0.6,
        threshold_loss_weight=0.5,
        ner_loss_weight=0.5,
    ):
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.cache_dir=cache_dir
        self.revision=revision
        self.use_auth_token=use_auth_token
        self.hidden_dropout_prob=hidden_dropout_prob
        self.max_span_width = max_span_width
        self.use_span_width_embedding = use_span_width_embedding
        self.linear_size = linear_size
        self.init_temperature = init_temperature
        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_loss_weight = span_loss_weight
        self.threshold_loss_weight = threshold_loss_weight
        self.ner_loss_weight = ner_loss_weight

        self.max_span_width = max_span_width
        self.use_span_width_embedding = use_span_width_embedding
        self.linear_size = linear_size
        self.init_temperature = init_temperature
        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_loss_weight = span_loss_weight
        self.threshold_loss_weight = threshold_loss_weight
        self.ner_loss_weight = ner_loss_weight


@dataclass
class EmptyConfig:
    pass

@dataclass
class TrainingConfig:
    run_name: str
    output_dir: str
    logging_dir: str


@dataclass
class DataConfig:
    tasks: dict
    eval_split: str = 'eval'
    label_format: Literal["index", "triangular_onehot"] = "index"
    data_path: str = "data/dataset"
    process_func: str = None
    predict_split: str | None = None
    full_data: bool = False
    dev: bool = False

@dataclass
class BaseConfig:
    model_config_class: ClassVar[Type[EmptyConfig]]
    data_config_class: ClassVar[Type[DataConfig]]
    training_config_class: ClassVar[Type[TrainingConfig]]
    run_name: str
    base_path: Path = Path("results/")

    @property
    def output_dir(self) -> Path:
        return self.base_path / self.run_name

    @property
    def logging_dir(self) -> Path:
        return self.output_dir / "logs"

    @classmethod
    def from_dict(cls, json_dict: dict) -> "BaseConfig":
        """Create config from a dictionary, handling nested dataclasses"""
        model_json = json_dict.pop("model", {})
        data_json = json_dict.pop("data", {})
        training_json = json_dict.pop("training", {})
        config = cls(**json_dict)

        model = cls.model_config_class(**model_json)
        data = cls.data_config_class(**data_json)
        training = cls.training_config_class(
            run_name=config.run_name,
            output_dir=config.output_dir,
            logging_dir=config.logging_dir,
            **training_json
        )

        config.model = model
        config.data = data
        config.training = training
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "BaseConfig":
        """Load config from a JSON file"""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Save config to a JSON file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
