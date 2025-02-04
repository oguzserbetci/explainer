import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import datasets
import polars as pl
from tqdm.auto import tqdm
from collections import defaultdict
import joblib

from train_baseline import BaselineConfig, load_labels

SPLIT = "test"
IDENTIFIER = "_id"

if __name__ == "__main__":
    config = BaselineConfig.from_json(os.path.abspath(sys.argv[-1]))

    log_path = Path(f'{config.logging_dir}/{SPLIT}_baseline_{datetime.today().strftime("%Y-%m-%d")}.log')
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger("transformers")
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    dataset = datasets.load_from_disk(config.data.data_path)[SPLIT]

    with open(config.output_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = joblib.load(f)
    X = vectorizer.transform(dataset["text"])
    feature_names = vectorizer.get_feature_names_out()

    classifiers = {}
    for task in config.data.tasks.keys():
        with open(config.output_dir / f"{task}.pkl", "rb") as f:
            clf = pickle.load(f)
        classifiers[task] = clf

    X_words = vectorizer.inverse_transform(X)

    counter = Counter()
    for sample in X_words:
        counter.update(list(set(sample.tolist())))

    def get_individual_attrs(X, y, preds, dataset, clf, feature_names, target_names, task):
        coef = clf.coef_
        if coef.shape[0] == 1:
            coef = np.vstack([-1 * coef, coef])

        attrs = []
        for sample_repr, sample, pred, label in tqdm(zip(X, dataset, preds, y)):
            feature_effects = sample_repr.multiply(coef[[pred]])
            feature_mask = feature_effects.nonzero()[1]
            _feature_names = feature_names[feature_mask]
            feature_effects = feature_effects.toarray()[0, feature_mask]
            for term, attr in zip(_feature_names, feature_effects):
                attrs.append(
                    dict(
                        task=task,
                        attr=attr,
                        term=term,
                        type=f'{term.count(" ") + 1}gram',
                        _id=sample[IDENTIFIER],
                        pred=pred,
                        label=label,
                        index=pred,
                    )
                )
        return attrs

    os.makedirs(config.output_dir / "plots", exist_ok=True)

    def coef_analysis(clf, x, feature_names, target_names_simplified):
        feature_counts = np.array((X != 0).sum(0).tolist())[0]
        # learned coefficients weighted by frequency of appearance
        average_feature_effects = clf.coef_ * np.asarray(X.mean(axis=0)).ravel()
        if average_feature_effects.shape[0] == 1:
            average_feature_effects = np.vstack(
                [
                    -1 * average_feature_effects,
                    average_feature_effects,
                ]
            )
        for i, label in enumerate(target_names):
            sorted_feature_effects = np.argsort(average_feature_effects[i])
            top = sorted_feature_effects[::-1]
            if i == 0:
                top_indices = top
                top_coef = average_feature_effects[i][top]
            else:
                top_indices = np.vstack([top_indices, top])
                top_coef = np.vstack([top_coef, average_feature_effects[i][top]])

        row = defaultdict(list)
        for i, (indices, coefs) in enumerate(zip(top_indices, top_coef)):
            row["term"].extend(feature_names[indices].tolist())
            row["occurance"].extend(feature_counts[indices].tolist())
            row["attr"].extend(coefs.tolist())
            row["pred"].extend([i] * len(coefs))
        row["task"].extend([task] * len(row["attr"]))
        return row

    rows = []
    attrs = []
    for task, clf in classifiers.items():
        y, target_names = load_labels(config, dataset, task)
        target_names_simplified = [tn for tn in target_names if re.search(r"=[012]", tn)]

        preds = clf.predict(X)
        attrs.extend(
            get_individual_attrs(
                X,
                y,
                preds,
                dataset,
                clf,
                feature_names,
                target_names_simplified,
                task,
            )
        )

        coefs = coef_analysis(clf, X, feature_names, target_names_simplified)
        rows.append(coefs)

    attr_df = pl.DataFrame(
        attrs,
        schema_overrides={
            "term": pl.Categorical("lexical"),
            "type": pl.Categorical("lexical"),
            "task": pl.Categorical("lexical"),
            "_id": pl.Categorical("lexical"),
        },
    )
    (config.output_dir / "tfidf_attrs" / SPLIT).mkdir(parents=True, exist_ok=True)
    attr_df.write_parquet(config.output_dir / "tfidf_attrs" / SPLIT / "attr_processed.parquet")

    df = pl.concat([pl.DataFrame(row) for row in rows])
    df = df.filter(pl.col("attr") != 0)
    df.write_csv(config.output_dir / "tfidf_naive.csv")
