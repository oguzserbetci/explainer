import logging
from pathlib import Path
from somajo import SoMaJo

import click
import polars as pl
from datasets import DatasetDict
from tqdm.auto import tqdm

from src.attribution_processing import get_attributions, get_sentences
from train_transformers import TransformersConfig
from transformers import AutoTokenizer

IDENTIFIER = "_id"


@click.command()
@click.option("--config_path", "-c", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--split", "-s", default="train")
@click.option("--methods", "-m", default=["lrp", "ig"], multiple=True)
@click.option("--language", "-l", default="en")  # en_PTB for english
@click.option("--force", default=False, is_flag=True)  # en_PTB for english
@click.argument("checkpoint_path", type=click.Path(exists=True, path_type=Path))  # Path to the parquet files.
def main(checkpoint_path, config_path, split, methods, force, language):
    config = TransformersConfig.from_json(config_path)
    dataset = DatasetDict.load_from_disk(config.data.data_path)[split]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if language == "en":
        sentence_model = SoMaJo("en_PTB")
    elif language == "de":
        sentence_model = SoMaJo("de_CMC")
    else:
        raise ValueError(f"Language {language} not supported")

    id2text = {str(sample[IDENTIFIER]): sample["text"] for sample in dataset}
    for method in methods:
        attr_path = checkpoint_path / f"{method}_attrs/{split}"
        for file_path in tqdm(list(sorted(attr_path.glob("*.parquet")))):
            if file_path.stem.endswith("_processed"):
                continue

            out_path = file_path.with_stem(f"{file_path.stem}_processed")
            if not force and out_path.exists():
                continue

            word_attrs = pl.read_parquet(file_path)
            sample_id = file_path.stem

            sentences = get_sentences(id2text[sample_id], tokenizer, sentence_model)
            logging.info(sample_id)
            logging.info(id2text)
            logging.info(sentences)

            dfs = []
            for (task, index, label, pred, proba), group_df in word_attrs.group_by(
                ["task", "index", "label", "pred", "proba"], maintain_order=True
            ):
                terms_attrs = group_df.select(["term", "attr"]).to_dict(as_series=False)
                terms, attrs, types = get_attributions(
                    zip(terms_attrs["term"], terms_attrs["attr"]), sentences, language
                )

                df = pl.from_dict({"term": terms, "attr": attrs, "type": types})
                df = df.with_columns(
                    pl.lit(sample_id).alias("_id"),
                    pl.lit(label).alias("label"),
                    pl.lit(index).alias("index"),
                    pl.lit(task).alias("task"),
                    pl.lit(pred).alias("pred"),
                    pl.lit(proba).alias("proba"),
                )
                # df = df.with_columns(
                #     ((pl.col("attr") - pl.col("attr").mean().over("type")) / pl.col("attr").std().over("type")).alias(
                #         "attr_normed"
                #     )
                # )
                # df = df.with_columns((pl.col("attr") / pl.col("attr").abs().max().over("type")).alias("attr_scaled"))
                # df = df.with_columns(
                #     (pl.col("attr") / (pl.col("attr").abs().sum().over("type"))).alias("attr_percentile")
                # )
                dfs.append(df)

            if dfs:
                logging.info(f"Writing to {out_path}")
                pl.concat(dfs).write_parquet(out_path)


if __name__ == "__main__":
    main()
