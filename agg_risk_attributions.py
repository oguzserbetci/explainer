from pathlib import Path

import click
import polars as pl
from datasets import DatasetDict
from sklearn.metrics import f1_score
import logging
import time
from src.attribution_aggregation import avg_attr, homogeniety_weighted_attr
from agg_attributions import get_f1score, read_attr_data

@click.command()
@click.option("--dataset_path", "-d", default="data/dataset_chunked_befund")
@click.option("--method", "-m", default="ig")
@click.option("--split", default="train")
@click.option("--with_neg", default=True, is_flag=True)  # By default without negatives.
@click.option("--over", "-o", default=["task", "index", "label", "pred"], multiple=True, type=str)
@click.option("--top_k", "-k", default=50)
@click.argument("checkpoint_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(dataset_path, checkpoint_path, method, split, with_neg, over, top_k):
    over = list(over)

    # Load attributions
    attr_folder = f"{method}_attrs_new/"
    base_path_name = f'{split}{"_pos" if with_neg else ""}'
    if not (checkpoint_path / attr_folder / f"{base_path_name}_df.parquet").exists():
        start_time = time.time()
        attr_parquet_paths = list((checkpoint_path / attr_folder / split).glob("*_processed.parquet"))
        df = read_attr_data(attr_parquet_paths, filter_pos=with_neg)
        df.write_parquet(checkpoint_path / attr_folder / f"{base_path_name}_df.parquet")
        logging.info(
            f"+++ {checkpoint_path / attr_folder / f'{base_path_name}_df.parquet'} loaded and cached in {time.time() - start_time}"
        )
    else:
        df = pl.read_parquet(checkpoint_path / attr_folder / f"{base_path_name}_df.parquet")

    df = df.filter((pl.col("label") < pl.col("pred")) & (pl.col('index') == pl.col('pred')))

    valid_tasks = df.select(pl.col("task").cat.get_categories()).filter(
        ~pl.col("task").str.contains("=") & ~pl.col("task").str.contains("-")
    )

    # Aggregate attributions
    base_path_name = f'{split}{"_pos" if with_neg else ""}_{"-".join(over)}'
    # Homogeneity weighted attr
    if not (checkpoint_path / attr_folder / f"{base_path_name}_homogeneity_risk_df.parquet").exists():
        homogeneity_df = homogeniety_weighted_attr(df.filter(pl.col("task").is_in(valid_tasks)), over=over)
        homogeneity_df.write_parquet(checkpoint_path / attr_folder / f"{base_path_name}_homogeneity_risk_df.parquet")
    else:
        homogeneity_df = pl.read_parquet(checkpoint_path / attr_folder / f"{base_path_name}_homogeneity_risk_df.parquet")

    logging.info(homogeneity_df
        .select(pl.all().top_k_by("attrs", k=20).over(over, mapping_strategy="explode"))
        .sort(["task", "label", "attr"])
        .with_columns(pl.col("terms").cast(pl.String))
        .group_by(["task", "label", "pred"], maintain_order=True)
        .agg(pl.format("{} ({})", pl.col("terms"), pl.col("occurance")).str.concat(", "))
        .to_pandas()
        .to_latex(index=False)
        )



if __name__ == "__main__":
    main()
