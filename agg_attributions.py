from pathlib import Path

import click
import polars as pl
from sklearn.metrics import f1_score
import logging
import time
from src.attribution_aggregation import avg_attr, homogeniety_weighted_attr, sum_sqrt_attr
import plotly.express as px


ATTR_FIELD = "attr"
AGG_METHODS = {
    "avg": avg_attr,
    "sum_sqrt": sum_sqrt_attr,
    "homogeneity": homogeniety_weighted_attr,
}


def get_f1score(x):
    return f1_score(x[0], x[1], average="macro")


def read_attr_data(parquet_paths, attr_field, type="!sentence", filter_ids=None, filter_pos=False):
    attr_df = ( pl.scan_parquet(parquet_paths)
        .cast(
            {"task": pl.Categorical("lexical"), "_id": pl.Categorical("lexical"),# 'term': pl.Categorical("lexical")
            },
        )
    )
    if isinstance(attr_df.collect_schema()['term'], pl.String):
        attr_df = attr_df.with_columns(pl.col("term").str.to_lowercase().cast(pl.Categorical("lexical")))

    attr_df = attr_df.filter(~pl.col("task").cast(pl.String).str.contains("=|-"))

    if filter_ids is not None:
        attr_df = attr_df.filter(pl.col("_id").is_in(filter_ids))

    if "!" in type:
        attr_df = attr_df.filter(pl.col("type") != type[1:])
    else:
        attr_df = attr_df.filter(pl.col("type") == type)

    attr_df = attr_df.with_columns((pl.col(attr_field) >= 0).alias("pos"))

    # Create histogram
    fig = px.scatter(
        attr_df.filter(pl.col("type") == "1gram")
        .group_by("_id", "pos")
        .agg(pl.col(attr_field).sum(), pl.col(attr_field).count().alias("nattr"))
        .collect()
        .to_pandas(),
        x=attr_field,
        y="nattr",
        color="pos",
        title="Distribution of Attribution Values over samples",
    )
    # Update layout
    fig.update_layout(xaxis_title="Attribution sum", yaxis_title="Number of attribtions", bargap=0.1)

    # Save to file
    fig.write_image(parquet_paths[0].parent.parent / "attributions.png")

    if filter_pos:
        attr_df = attr_df.filter(pl.col("pos"))
    
    return (
        attr_df.group_by(["term", "type", "pos", "index", "label", "pred", "task"]) .agg(pl.col(attr_field).abs().sum(), pl.col(attr_field).count().alias("occurance")) .collect()
    )


def get_top_terms(agg_df, attr_field, over=["task"], k=50):
    return agg_df.sort(attr_field, descending=True).group_by(over, maintain_order=True).head(k)


def get_exclusive_top_terms(agg_df, attr_field, over=["task"], k=50):
    agg_df = agg_df.group_by("term").agg(
        *(pl.col(o).get(pl.col(attr_field).arg_max()) for o in over),
        pl.col("occurance").get(pl.col(attr_field).arg_max()),
        pl.col(attr_field).max(),
    )
    return agg_df.sort(attr_field, descending=True).group_by(over, maintain_order=True).head(k)


@click.command()
@click.option("--explanation_methods", "-e", default=["ig", "lrp"], multiple=True)
@click.option("--aggregation_methods", "-a", default=["homogeneity", "avg"], multiple=True)
@click.option("--split", "-s", default="train")
@click.option("--with_neg", default=True, is_flag=True)  # By default without negatives.
@click.option("--type", "-t", default="!sentence", type=str)
@click.option("--over", "-o", default=["task"], multiple=True, type=str)
@click.option("--included_preds", "-p", type=click.Choice(["correct", "wrong", "all", "higher"]), default="all")
@click.option("--top_k", "-k", default=50)
@click.option("--force", "-f", is_flag=True, default=False)
@click.argument("checkpoint_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(checkpoint_path, explanation_methods, aggregation_methods, split, type, included_preds, with_neg, over, top_k, force):
    over = list(over)

    # Load attributions
    for method in explanation_methods:
        attr_folder = f"{method}_attrs/"
        base_path_name = f'{split}_{ATTR_FIELD}{"_pos" if with_neg else ""}_{type}'
        if force or not (checkpoint_path / attr_folder / f"{base_path_name}_df.parquet").exists():
            start_time = time.time()
            attr_parquet_paths = list((checkpoint_path / attr_folder / split).glob("*_processed.parquet"))[:500]
            if not attr_parquet_paths:
                raise ValueError("No attribtion files found")
            print(f"Reading from {len(attr_parquet_paths)} files.")
            df = read_attr_data(attr_parquet_paths, attr_field=ATTR_FIELD, filter_pos=with_neg)
            df.write_parquet(checkpoint_path / attr_folder / f"{base_path_name}_df.parquet")
            logging.info(
                f"+++ {checkpoint_path / attr_folder / f'{base_path_name}_df.parquet'} loaded and cached in {time.time() - start_time}"
            )
        else:
            df = pl.read_parquet(checkpoint_path / attr_folder / f"{base_path_name}_df.parquet")
        
        if included_preds == "wrong":
            df = df.filter(pl.col("label") != pl.col("pred"))
        elif included_preds == "correct":
            df = df.filter(pl.col("label") == pl.col("pred"))
        elif included_preds == "higher":
            df = df.filter(pl.col("label") < pl.col("pred"))
        elif included_preds == "all":
            pass
        
        for agg_method in aggregation_methods:
            print(f'Calculating {agg_method} aggregation for {method} attributions.')
            # Aggregate attributions
            base_path_name = (
                f'{split}_{ATTR_FIELD}{"_pos" if with_neg else ""}_{type}_{included_preds}_{"-".join(over)}'
            )
            # Homogeneity weighted attr
            if force or not (checkpoint_path / attr_folder / f"{base_path_name}_{agg_method}_df.parquet").exists():
                agg_df = AGG_METHODS[agg_method](df, attr_field=ATTR_FIELD, over=over)
                agg_df.write_parquet(checkpoint_path / attr_folder / f"{base_path_name}_{agg_method}_df.parquet")
            else:
                agg_df = pl.read_parquet(
                    checkpoint_path / attr_folder / f"{base_path_name}_{agg_method}_df.parquet"
                )
            
            agg_top_df = get_top_terms(agg_df, ATTR_FIELD, over, top_k).select(
                ["term"] + over + [ATTR_FIELD, "occurance"]
            )
            assert agg_top_df[ATTR_FIELD].is_not_null().all(), breakpoint()
            agg_top_df.write_csv(checkpoint_path / attr_folder / f"{base_path_name}_{agg_method}_top_{top_k}.csv")

            agg_exclusive_top_df = get_exclusive_top_terms(agg_df, ATTR_FIELD, over, top_k).select(
                ["term"] + over + [ATTR_FIELD, "occurance"]
            )
            agg_exclusive_top_df.write_csv(
                checkpoint_path / attr_folder / f"{base_path_name}_{agg_method}_exclusive_top_{top_k}.csv"
            )

            logging.info(f"+++ {base_path_name} {agg_method} agg attr TOP {top_k} phrases")
            logging.info(agg_top_df)


if __name__ == "__main__":
    main()
