from pathlib import Path

import click
import numpy as np
import polars as pl
import torch
from datasets import DatasetDict
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.explainer import SequenceClassificationExplainer, get_lrp_ranks
from src.models import pred_for_tasks, BertForLongerSequenceClassification
from src.models_lrp import BertForLongerSequenceClassificationXAI
from train_transformers import TransformersConfig
import src.data_processing as data_processing  # noqa: F401

IDENTIFIER = "_id"


# %%
@click.command()
@click.option("--device", "-d", default="cuda")
@click.option("--config_path", "-c", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--split", default="train")
@click.option("--n_samples", "-n", default=None, type=int)
@click.option("--methods", "-m", default=["lrp", 'ig'], multiple=True)
@click.option("--force", default=False, is_flag=True)
@click.argument("checkpoint_path", type=click.Path(exists=True, path_type=Path))
def main(checkpoint_path, config_path, split, n_samples, device, force, methods):
    config = TransformersConfig.from_json(config_path)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    dataset = DatasetDict.load_from_disk(config.data.data_path)[split]
    
    if config.data.process_func:
        dataset = dataset.map(eval(config.data.process_func), fn_kwargs={'tokenizer': tokenizer})

    for method in methods:
        collect_attributions(
            dataset, tokenizer, checkpoint_path, method, split, n_samples, device, force
        )

def collect_attributions(dataset, tokenizer, checkpoint_path, method, split, n_samples, device, force):
    outdir = checkpoint_path / f"{method}_attrs/{split}/"
    outdir.mkdir(parents=True, exist_ok=True)

    if method == "lrp":
        model = BertForLongerSequenceClassificationXAI.from_pretrained(checkpoint_path).to(device)
        attr_method = collect_lrp_attributions
    elif method == "ig":
        model = BertForLongerSequenceClassification.from_pretrained(checkpoint_path).to(device)
        attr_method = collect_ig_attributions
    else:
        raise NotImplementedError

    n = 0
    for sample in tqdm(dataset, total=len(dataset)):
        outpath = outdir / f"{sample[IDENTIFIER]}.parquet"
        if not force and outpath.exists():
            continue

        if len(sample["input_ids"]) > 5:
            continue

        if len(torch.Tensor(sample['input_ids']).shape) == 1:
            sample['input_ids'] = [sample['input_ids']]
            sample['token_type_ids'] = [sample['token_type_ids']]
            sample['attention_mask'] = [sample['attention_mask']]

        try:
            result = attr_method(sample, model, tokenizer)
        except Exception as e:
            print("Error calculating attributions:", e)
            continue

        n += 1
        result.write_parquet(outpath)
        if n == n_samples:
            break


def get_term_attributions(
    sample,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    token_type_ids,
    logit_function,
):
    tokens = np.array([tok for seq in input_ids for tok in tokenizer.convert_ids_to_tokens(seq)])
    mask_condition = np.where(~np.isin(input_ids.cpu().flatten().numpy(), tokenizer.all_special_ids))[0]
    extended_attention_mask: torch.Tensor = model.bert.get_extended_attention_mask(attention_mask, input_ids.shape)

    relevance, logits = get_lrp_ranks(
        model=model,
        tokenizer=tokenizer,
        sample=sample,
        output=None,
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": extended_attention_mask,
            "token_type_ids": token_type_ids,
        },
        logit_function=logit_function,
    )
    relevance = relevance[mask_condition]
    tokens = tokens[mask_condition]
    assert len(relevance) == len(tokens)
    return list(zip(tokens, relevance))


def collect_lrp_attributions(sample, model, tokenizer):
    dfs = []

    # Run prediction
    input_ids = torch.LongTensor(sample["input_ids"]).to(model.device)
    attention_mask = torch.LongTensor(sample["attention_mask"]).to(model.device)
    token_type_ids = torch.LongTensor(sample["token_type_ids"]).to(model.device)
    overflow_to_sample_mapping = torch.zeros(len(input_ids), dtype=torch.long).to(model.device)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            overflow_to_sample_mapping=overflow_to_sample_mapping,
        )
    preds = pred_for_tasks(model.config.tasks, output.logits)

    for (task, num_labels), pred, logit_offset in zip(
        model.config.tasks.items(),
        preds,
        [0] + np.cumsum(list(model.config.tasks.values())).tolist(),
    ):
        torch.cuda.empty_cache()

        # attribution for task - rest logits
        task_label = sample[task]
        attr_index = logit_offset + pred
        logit_function = lambda x: x[[attr_index]]
        proba_function = lambda logits: torch.softmax(
            logits[logit_offset : logit_offset + model.config.tasks[task]], -1
        )[pred]

        word_attributions = get_term_attributions(
            sample,
            model,
            tokenizer,
            input_ids,
            attention_mask,
            token_type_ids,
            logit_function,
        )

        df = pl.from_dicts(
            {"term": w, "attr": a} for w, a in word_attributions if w not in ["[CLS]", "[SEP]", "[PAD]"]
        )
        df = df.with_columns(
            pl.lit(pred).alias("pred"),
            pl.lit(task_label).alias("label"),
            pl.lit(pred).alias("index"),
            pl.lit(task).alias("task"),
            pl.lit(proba_function(output.logits[0])).alias("proba"),
        )
        dfs.append(df)

    return pl.concat(dfs)


def collect_ig_attributions(sample, model, tokenizer):
    dfs = []
    input_ids = torch.LongTensor(sample["input_ids"]).to(model.device)
    attention_mask = torch.LongTensor(sample["attention_mask"]).to(model.device)
    token_type_ids = torch.LongTensor(sample["token_type_ids"]).to(model.device)
    overflow_to_sample_mapping = torch.zeros(len(input_ids), dtype=torch.long).to(model.device)
    output = model( input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, )
    preds = pred_for_tasks(model.config.tasks, output.logits)
    for (task, num_labels), pred, logit_offset in zip(
        model.config.tasks.items(), preds, [0] + np.cumsum(list(model.config.tasks.values())).tolist()
    ):
        # label attributions
        cls_explainer = SequenceClassificationExplainer(
            model,
            tokenizer,
        )
        task_label = sample[task]
        word_attributions = cls_explainer(
            input_ids,
            attention_mask,
            index=logit_offset + pred,
            n_steps = 50
        )

        df = pl.from_dicts({"term": w, "attr": a} for w, a in word_attributions if w not in ["[CLS]", "[SEP]", "[PAD]"])
        df = df.with_columns(
            pl.lit(pred).alias("pred"),
            pl.lit(task_label).alias("label"),
            pl.lit(pred).alias("index"),
            pl.lit(task).alias("task"),
            pl.lit(torch.softmax(output.logits[0, logit_offset : logit_offset + num_labels], -1)[pred]).alias("proba"),
        )
        dfs.append(df)

    return pl.concat(dfs)


if __name__ == "__main__":
    main()
