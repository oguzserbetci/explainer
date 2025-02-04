from transformers import DataCollatorWithPadding
from typing import Any
import numpy as np
from pathlib import Path
import json


class DataCollatorForMultiSentenceMultitask(DataCollatorWithPadding):
    def __init__(
        self,
        *args,
        label_format="index",
        tasks=None,
        adversary_tasks=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_format = label_format
        self.tasks = tasks
        self.adversary_tasks = adversary_tasks

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        flat_samples = []
        for i, sample in enumerate(features):
            adversary_labels = None
            if self.label_format == "triangular_onehot":
                labels = np.hstack(
                    [
                        np.tril(np.ones(num_labels + 1))[sample[task]]
                        for task, num_labels in self.tasks.items()
                    ]
                )
                if self.adversary_tasks is not None:
                    adversary_labels = np.hstack(
                        [
                            np.tril(np.ones(num_labels + 1))[sample[task]]
                            for task, num_labels in self.adversary_tasks.items()
                        ]
                    )
            elif self.label_format == "index":
                labels = np.array([sample[task] for task in self.tasks.keys()])
                if self.adversary_tasks is not None:
                    adversary_labels = np.array(
                        [sample[task] for task in self.adversary_tasks.keys()]
                    )
            else:
                raise NotImplementedError

            if hasattr(sample["input_ids"][0], "__iter__"):
                for input_ids, token_type_ids, attention_mask in zip(
                    sample["input_ids"], sample["token_type_ids"], sample["attention_mask"]
                ):
                    _sample = {
                        "input_ids": input_ids,
                        "token_type_ids": token_type_ids,
                        "attention_mask": attention_mask,
                        "overflow_to_sample_mapping": i,
                        "labels": labels,
                    }
                    if adversary_labels is not None:
                        _sample["adversary_labels"] = adversary_labels
                    flat_samples.append(_sample)
            else:
                _sample = {
                    "input_ids": sample["input_ids"],
                    "token_type_ids": sample["token_type_ids"],
                    "attention_mask": sample["attention_mask"],
                    "labels": labels,
                }
                if adversary_labels is not None:
                    _sample["adversary_labels"] = adversary_labels
                flat_samples.append(_sample)

        return super().__call__(flat_samples)


# class DataCollatorForMultitask(DataCollatorWithPadding):
#     def __init__(self, *args, label_format="triangular_onehot", tasks=None, adversary_tasks=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.label_format = label_format
#         self.tasks = tasks
#         self.adversary_tasks = adversary_tasks

#     def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
#         samples = []
#         for i, sample in enumerate(features):
#             input_ids, token_type_ids, attention_mask = sample['input_ids'], sample['token_type_ids'], sample['attention_mask']
#             adversary_labels = None
#             if self.label_format == 'triangular_onehot':
#                 labels = np.hstack([np.tril(np.ones(num_labels + 1))[sample[task], 1:] for task, num_labels in self.tasks.items()])
#                 if self.adversary_tasks is not None:
#                     adversary_labels = np.hstack([np.tril(np.ones(num_labels + 1))[sample[task], 1:] for task, num_labels in self.adversary_tasks.items()])
#             elif self.label_format == 'index':
#                 labels = np.array([sample[task] for task in self.tasks.keys()])
#                 if self.adversary_tasks is not None:
#                     adversary_labels = np.array([sample[task] for task in self.adversary_tasks.keys()])
#             else:
#                 raise NotImplementedError
#             _sample = {
#                 'input_ids': input_ids,
#                 'token_type_ids': token_type_ids,
#                 'attention_mask': attention_mask,
#                 'overflow_to_sample_mapping': i,
#                 'labels': labels,
#             }
#             if adversary_labels is not None:
#                 _sample['adversary_labels'] = adversary_labels
#             samples.append(_sample)
#         return super().__call__(samples)
