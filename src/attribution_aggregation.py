import polars as pl


def avg_attr(df, attr_field, over=["task"]):
    df_ = (
        df.filter(pl.col(attr_field).abs() > 1e-5)
        .group_by(["term"] + over)
        .agg((pl.col(attr_field).abs().sum() / pl.col("occurance").sum()).alias(attr_field), pl.col("occurance").sum())
    )

    return df_


def sum_sqrt_attr(df, attr_field, over=["task"]):
    df_ = df.group_by(["term"] + over).agg(
        (pl.col(attr_field).abs().sum().sqrt()).alias(attr_field),
        pl.col("occurance").sum(),
    )
    return df_


def homogeniety_weighted_attr(df, attr_field, over=["task"]):
    _pcj = df.group_by(["term"] + over).agg(pl.col(attr_field).abs().sum().sqrt(), pl.col("occurance").sum())
    pcj = _pcj.with_columns((pl.col(attr_field) / pl.col(attr_field).abs().sum().over("term")))
    hj = (
        pcj.with_columns((pl.col(attr_field) * pl.col(attr_field).log()).alias(attr_field))
        .group_by("term")
        .agg(pl.col(attr_field).sum() * -1)
    )
    H_weight = hj.with_columns(
        (
            1
            - ((pl.col(attr_field) - pl.col(attr_field).min()) / (pl.col(attr_field).max() - pl.col(attr_field).min()))
        ).alias(attr_field)
    )
    iHcj = (
        _pcj.join(H_weight, on="term", how="inner", suffix="_homogeneity_weight")
        .with_columns((pl.col(attr_field) * pl.col(f"{attr_field}_homogeneity_weight")).alias(attr_field))
        .drop(f"{attr_field}_homogeneity_weight")
    )
    return iHcj
