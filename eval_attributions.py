import os
from collections import Counter
from pathlib import Path

import click
import numpy as np
import plotly.express as px
import polars as pl
import torch
from datasets import DatasetDict
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from train_transformers import TransformersConfig

from src.explainer import SequenceClassificationExplainer, get_lrp_ranks
from src.models import BertForLongerSequenceClassification, logits_for_tasks, pred_for_tasks
from src.models_lrp import BertForLongerSequenceClassificationXAI
import src.data_processing as data_processing  # noqa: F401


def get_ig_ranks(model, tokenizer, sample, task, **kwargs):
    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer,
    )
    cls_explainer.n_steps = 20
    word_attributions = cls_explainer(
        torch.LongTensor(sample["input_ids"]).to(model.device),
        torch.LongTensor(sample["attention_mask"]).to(model.device),
        index=sample[task],
    )

    _, attrs = zip(*word_attributions)
    return np.array(attrs)


def get_lgxa_ranks(model, tokenizer, sample, task, **kwargs):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer, attribution_type="lgxa")
    word_attributions = cls_explainer(
        torch.LongTensor(sample["input_ids"]).to(model.device),
        torch.LongTensor(sample["attention_mask"]).to(model.device),
        index=sample[task],
    )

    _, attrs = zip(*word_attributions)
    return np.array(attrs)


def get_random_ranks(model, tokenizer, sample, **kwargs):
    return np.random.rand(len(np.array(sample["input_ids"]).flatten()))


def get_attn_ranks(model, tokenizer, sample, output, **kwargs):
    return (
        torch.stack([a.detach() for a in output.attentions])
        .mean(0)[:, -1, :, 0]
        .cpu()
        .numpy()
        .flatten()
    )


@click.command()
@click.option("--n_samples", "-N", default=10)
@click.option("--device", "-d", default="cuda")
@click.option("--task", "-t", default="sentiment")
@click.option("--config_path", "-c", default=None, type=click.Path(exists=True, path_type=Path))
@click.argument("checkpoint_path", type=click.Path(exists=True, path_type=Path))
def main(config_path, checkpoint_path, n_samples, device, task):
    config = TransformersConfig.from_json(config_path)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    dataset = DatasetDict.load_from_disk(config.data.data_path)

    eval_set = dataset["eval"]
    if config.data.process_func:
        eval_set = eval_set.map(eval(config.data.process_func), fn_kwargs={"tokenizer": tokenizer})

    results = []
    pbar = tqdm(
        enumerate(
            [
                get_lrp_ranks,
                get_ig_ranks,
                get_random_ranks,
                get_attn_ranks,
                get_lgxa_ranks,
            ]
        ),
        position=0,
    )
    for i, method in pbar:
        pbar.set_description(method.__name__)
        if method.__name__ == "get_lrp_ranks":
            model = BertForLongerSequenceClassificationXAI.from_pretrained(checkpoint_path).to(device).eval()
        else:
            model = BertForLongerSequenceClassification.from_pretrained(checkpoint_path).to(device).eval()

        pbar2 = tqdm(eval_set, position=1, total=n_samples)
        _ids = []
        for sample in pbar2:
            input_ids = torch.LongTensor(sample["input_ids"]).to(model.device)
            attention_mask = torch.LongTensor(sample["attention_mask"]).to(model.device)
            token_type_ids = torch.LongTensor(sample["token_type_ids"]).to(model.device)
            if len(input_ids) > 5:
                continue

            torch.cuda.empty_cache()
            try:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                )
                task_index = list(model.config.tasks.keys()).index(task)
                task_pred = pred_for_tasks(model.config.tasks, output.logits)[task_index]

                _ids.append(sample["_id"])
                proba_function = lambda logits: logits_for_tasks(model.config.tasks, logits)[task_index][0][task_pred]  # noqa: E731
                results.append(
                    (
                        method.__name__,
                        sample["_id"],
                        sample[task],
                        task_pred,
                        proba_function(output.logits),
                        0,
                        0,
                    )
                )

                extended_attention_mask: torch.Tensor = model.bert.get_extended_attention_mask(
                    attention_mask, input_ids.shape
                )
                attrs = method(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    output=output,
                    task=task,
                    model_inputs={
                        "input_ids": input_ids,
                        "attention_mask": extended_attention_mask,
                        "token_type_ids": token_type_ids,
                    },
                    logit_function=lambda x: x[sample[task]],
                )
            except (torch.OutOfMemoryError, TypeError) as error:
                print(error)
                continue

            if isinstance(attrs, tuple):
                attrs = attrs[0]
            ignored_tokens = np.where(
                np.isin(input_ids.cpu().flatten().numpy(), tokenizer.all_special_ids)
            )[0]
            attr_ranking = np.argsort(attrs)[::-1]
            attr_ranking = attr_ranking[np.where(~np.isin(attr_ranking, ignored_tokens))[0]]
            percentiles = list(range(1, 20, 1)) + list(range(20, 46, 5)) + list(range(50, 101, 10))
            masked_indices = [
                (attr_ranking[: int(idx)], percentile)
                for idx, percentile in zip(
                    np.percentile(np.arange(len(attr_ranking)), percentiles),
                    percentiles,
                )
            ]
            for masked_ind, percentile in tqdm(masked_indices, position=2, desc="masking"):
                with torch.no_grad():
                    masked_input = (
                        input_ids.flatten()
                        .index_fill(
                            0,
                            torch.LongTensor(masked_ind).to(input_ids.device),
                            tokenizer.mask_token_id,
                        )
                        .reshape_as(input_ids)
                    )
                    _o = model(
                        input_ids=masked_input,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    results.append(
                        (
                            method.__name__,
                            sample["_id"],
                            sample[task],
                            _o.logits[:, :3].argmax(-1).item(),
                            _o.logits[:, sample[task]].item(),
                            percentile,
                            len(masked_ind),
                        )
                    )

            pbar2.update(1)
            if len(_ids) == n_samples:
                break

    df = pl.DataFrame(
        results,
        schema=[
            "method",
            "_id",
            "label",
            "pred",
            "confidence",
            "removed_percentile",
            "removed_tokens",
        ],
    )
    df.write_csv(checkpoint_path / f"attr_{n_samples}_removal.csv")

    def get_f1score(x):
        return f1_score(x[0], x[1], average="micro", labels=range(3))

    f1_df = df.group_by(["method", "removed_percentile"]).agg(
        pl.map_groups(exprs=["label", "pred"], function=get_f1score).alias("f1"),
        pl.col("removed_tokens").mean().alias("mean_removed_tokens"),
    )

    len(_ids), Counter(df.group_by("_id").agg(pl.col("label").first())["label"])

    plot = px.line(
        f1_df.to_pandas().sort_values(["method", "removed_percentile"]),
        y="f1",
        x="removed_percentile",
        color="method",
        hover_data=["mean_removed_tokens"],
        range_y=(0, 1),
        symbol="method",
    )

    plot.write_html(checkpoint_path / f"attr_{n_samples}_removal.html")


if __name__ == "__main__":
    main()
