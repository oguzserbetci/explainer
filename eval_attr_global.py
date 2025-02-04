import logging
import re
from pathlib import Path

import click
import polars as pl
import torch
import datasets
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer

import src.data_processing as data_processing  # noqa: F401
from src.models import (
    BertForLongerSequenceClassification,
    logits_for_tasks,
    pred_for_tasks,
)
from datetime import datetime
from train_transformers import TransformersConfig

IDENTIFIER = "_id"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"eval_attr_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)


def load_attr_df(path):
    top_df = pl.read_parquet(path).sort("attr", descending=True)
    if ("_id" in top_df) and (top_df["_id"].dtype == pl.String):
        top_df = top_df.with_columns(pl.col("_id").str.split(","))
        top_df = top_df.with_columns(pl.col("_id").cast(pl.List(pl.UInt32), strict=True))
    return top_df


def _add_pred_logits(df):
    return df.with_columns(pl.col("preds").first().over("_id", order_by="n_masked_terms")).with_columns(
        pl.struct(["logits", "preds"])
        .map_elements(
            lambda x: ([l[p] for l, p in zip(x["logits"], x["preds"])] if x["logits"] is not None else None),
            return_dtype=pl.List(pl.Float64),
        )
        .alias("pred_logits"),
    )


@torch.no_grad()
def global_masks(
    model,
    tokenizer,
    dataset,
    device,
    top_df,
    n_masked_terms_limit=None,
    n_masked_tokens_limit=None,
    tasks=["caregrade_label"],
    batch_size=32,
):
    """Removal evaluation for given top tokens dataframe

    Args:
        top_df (pl.DataFrame): Terms to be masked in the order of occurance in terms column.
        n_masked_terms_limit (int, optional): Number of terms to delete. Defaults to 20.
        n_masked_tokens_limit (_type_, optional): Number of tokens in total to delete from a document. Defaults to None.

    Yields:
        pl.DataFrame: DataFrame made up of rows of n terms masked and the corresponding logits
    """
    for task in tqdm(tasks, position=1):
        task_index = list(model.config.tasks.keys()).index(task)
        task_df = top_df.filter(pl.col("task") == task)
        pbar2 = tqdm(dataset, position=2, desc='Masking')
        maskings = []
        for sample in pbar2:
            text = tokenizer.convert_tokens_to_string(tokenizer.batch_decode(tokenizer(sample["text"])['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)).lower()
            labels = [sample[task] for task in model.config.tasks.keys()]

            if "pred" in task_df.columns:
                _task_df = task_df.filter(pl.col("pred") == labels[task_index])
            elif "label" in task_df.columns:
                _task_df = task_df.filter(pl.col("label") == labels[task_index])
            else:
                _task_df = task_df.clone()

            # Replace ngrams
            n_masked_terms = 0
            n_missed_terms = 0
            n_masked_tokens = 0
            masked_terms = []

            terms = _task_df["term"].to_list()
            if len(terms) < n_masked_terms_limit:
                raise ValueError("Not enough terms to mask")

            logging.debug(text)

            maskings.append((sample[IDENTIFIER], text, labels, n_masked_terms, n_missed_terms, n_masked_tokens, masked_terms))
            pattern = ""
            for ngram in terms:
                found_terms = None
                ngram = ngram.replace("*", r"\*")
                try:
                    re_pattern = ngram.replace(" ", r"\b(?:.{0,12}|\s)\b")
                    re_pattern = rf"\b{re_pattern}\b"
                    found_terms = re.search(re_pattern, text, flags=re.MULTILINE & re.IGNORECASE)

                    if not found_terms:
                        n_missed_terms += 1
                        continue
                    else:
                        if not pattern:
                            pattern = re_pattern
                        else:
                            pattern += "|" + re_pattern
                        pass
                except Exception as e:
                    print(e)
                    n_missed_terms += 1

                if found_terms:
                    logging.debug(f"Masked {n_masked_terms} '{ngram}' [{masked_terms}]")
                    masked_terms.append(ngram)
                    n_masked_terms += 1
                    text = re.sub(
                        pattern,
                        "[MASK]",
                        text,
                        flags=re.MULTILINE & re.IGNORECASE,
                    )
                    n_masked_tokens = text.count("[MASK]")
                    if (n_masked_tokens_limit is not None) and (n_masked_tokens > n_masked_tokens_limit):
                        break
                    maskings.append((sample[IDENTIFIER], text, labels, n_masked_terms, n_missed_terms, n_masked_tokens, masked_terms))
                    if n_masked_terms >= n_masked_terms_limit:
                        logging.info(
                            f"Finished a sample with {n_masked_terms} masked terms and {n_masked_tokens} masked tokens"
                        )
                        break
            else:
                logging.warning(
                    f"Skip example ({len(sample['text'].split(' '))} words) as n_masked_terms_limit terms couldn't be found. Masked terms {n_masked_terms}, missed_terms {n_missed_terms}",
                )
                continue
        
        _ids, texts, labels, n_masked_terms, n_missed_terms, n_masked_tokens, masked_terms = list(zip(*maskings))

        for batch_ind in trange(0, len(texts), 32, position=2, desc='Predicting'):
            model_input = tokenizer(
                texts[batch_ind:batch_ind + 32],
                truncation=True,
                return_overflowing_tokens=True,
                stride=50,
                return_tensors="pt",
                padding=True,
            ).to(device)
            output = model(**model_input)

            logits = logits_for_tasks(model.config.tasks, output.logits)
            logits = list(zip(*logits))
            preds = pred_for_tasks(model.config.tasks, output.logits)
            preds = list(zip(*preds))
            
            masking_outputs = pl.DataFrame(
                {
                    "n_masked_terms": n_masked_terms[batch_ind:batch_ind + 32],
                    "masked_terms": masked_terms[batch_ind:batch_ind + 32],
                    "n_missed_terms": n_missed_terms[batch_ind:batch_ind + 32],
                    "n_masked_tokens": n_masked_tokens[batch_ind:batch_ind + 32],
                    "logits": logits,
                    "preds": preds,
                    "labels": labels[batch_ind:batch_ind + 32],
                    "text": texts[batch_ind:batch_ind + 32],
                    "_id": _ids[batch_ind:batch_ind + 32],
                }
            )
            
            df = pl.DataFrame(masking_outputs).with_columns(
                pl.lit(task).alias("task"),
                pl.lit(task_index).alias("task_index"),
            )
            
            yield df


@click.command()
@click.option("--split", "-s", default="test")
@click.option("--device", "-d", default="cuda")
@click.option("--n_masked_terms", "-m", default=50)
@click.option("--number_of_samples", "-n", default=None, type=int)
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path))
@click.option("--tasks", "-t", type=str, multiple=True)
@click.argument("checkpoint_path", type=click.Path(exists=True, path_type=Path))
@click.argument("parquet_paths", type=click.Path(path_type=Path), nargs=-1)
def main(split, device, number_of_samples, n_masked_terms, config, tasks, checkpoint_path, parquet_paths):
    config = TransformersConfig.from_json(config)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = BertForLongerSequenceClassification.from_pretrained(checkpoint_path).to(device).eval()

    dataset = datasets.DatasetDict.load_from_disk(config.data.data_path)[split]

    if number_of_samples is not None:
        stratify_by_column = tasks[0]
        logging.info(f"Loaded {number_of_samples} with stratification on column {stratify_by_column} ({dataset.features[stratify_by_column].dtype})")
        if not isinstance(dataset.features[stratify_by_column], datasets.ClassLabel):
            dataset = dataset.class_encode_column(stratify_by_column)
        dataset = dataset.train_test_split(train_size=number_of_samples, stratify_by_column=stratify_by_column, seed=0)[
            "train"
        ]

    if config.data.process_func:
        dataset = dataset.map(eval(config.data.process_func), fn_kwargs={"tokenizer": tokenizer})

    for parquet_path in parquet_paths:
        parquet_path = checkpoint_path / parquet_path
        attr_df = load_attr_df(parquet_path)  # .filter(~pl.col("term").cast(pl.String).str.contains(" "))

        output = list(
            global_masks(
                model,
                tokenizer,
                dataset,
                device,
                attr_df,
                n_masked_terms_limit=n_masked_terms,
                tasks=tasks,
            )
        )
        top_terms_mask_eval_df = pl.concat(output)
        top_terms_mask_eval_df = top_terms_mask_eval_df.sort("_id", "task", "n_masked_terms")
        top_terms_mask_eval_df = _add_pred_logits(top_terms_mask_eval_df)
        top_terms_mask_eval_df.write_parquet(parquet_path.with_stem(f"{parquet_path.stem}_mask_eval"))


if __name__ == "__main__":
    main()
