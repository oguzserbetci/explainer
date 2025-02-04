import joblib
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pickle import dump
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datasets
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)

from src.configurator import BaseConfig, DataConfig, TrainingConfig


@dataclass
class BaselineModelConfig:
    stem = False
    lowercase = True
    ngram_range = (1, 3)
    max_df = 0.3
    min_df = 2
    sublinear_tf = False

    def __post_init__(self):
        if isinstance(self.ngram_range, list):
            object.__setattr__(self, "ngram_range", tuple(self.ngram_range))


@dataclass
class BaselineConfig(BaseConfig):
    model_config_class = BaselineModelConfig
    data_config_class = DataConfig
    training_config_class = TrainingConfig


def load_labels(config, dataset, task):
    # split target in a training set and a test set
    y = dataset.select_columns([task]).to_pandas().to_numpy().squeeze()

    # order of labels in `target_names` can be different from `categories`
    target_names = [f"{task}={label}" for label in range(config.data.tasks[task])]
    return y, target_names


def plot_feature_effects(X, clf, feature_names, target_names):
    average_feature_effects = clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
    if average_feature_effects.shape[0] == 1:
        average_feature_effects = np.vstack([
            -1*average_feature_effects,
            average_feature_effects,
        ])

    for i, label in enumerate(target_names):
        top5 = np.argsort(average_feature_effects[i])[-5:][::-1]
        if i == 0:
            top = pd.DataFrame(feature_names[top5], columns=[label])
            top_indices = top5
        else:
            top[label] = feature_names[top5]
            top_indices = np.concatenate((top_indices, top5), axis=None)
    top_indices = np.unique(top_indices)
    predictive_words = feature_names[top_indices]

    # plot feature effects
    bar_size = 0.25
    padding = 0.75
    y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(target_names):
        ax.barh(
            y_locs + (i - 2) * bar_size,
            average_feature_effects[i, top_indices],
            height=bar_size,
            label=label,
        )
    ax.set(
        yticks=y_locs,
        yticklabels=predictive_words,
        ylim=[
            0 - 4 * bar_size,
            len(top_indices) * (4 * bar_size + padding) - 4 * bar_size,
        ],
    )
    ax.legend(loc="lower right")

    return fig, predictive_words


if __name__ == "__main__":
    config = BaselineConfig.from_json(os.path.abspath(sys.argv[-1]))
    os.makedirs(config.output_dir / "plots", exist_ok=True)

    log_path = Path(f'{config.logging_dir}/train_baseline_{datetime.today().strftime("%Y-%m-%d")}.log')
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger(__file__)
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    dataset = datasets.load_from_disk(config.data.data_path)
    dataset = dataset.filter(lambda x: x["text"] is not None)

    data_train = dataset["train"]
    data_test = dataset[config.data.eval_split]
    german_stopwords = set(stopwords.words("german"))

    def load_dataset(data_train, data_test, verbose=False, remove=()):
        """Load and vectorize the MD dataset."""

        # Extracting features from the training data using a sparse vectorizer
        t0 = time()
        vectorizer = TfidfVectorizer(
            ngram_range=config.model.ngram_range,
            lowercase=config.model.lowercase,
            sublinear_tf=config.model.sublinear_tf,
            max_df=config.model.max_df,
            min_df=config.model.min_df,
            stop_words=list(german_stopwords),
            analyzer="word",
        )
        X_train = vectorizer.fit_transform(data_train["text"])

        # Extracting features from the test data using the same vectorizer
        t0 = time()
        X_test = vectorizer.transform(data_test["text"])

        feature_names = vectorizer.get_feature_names_out()

        if verbose:
            # compute size of loaded data
            logger.info(f"{len(data_train['text'])} documents in train split")
            logger.info(f"{len(data_test['text'])} documents in eval split")
            logger.info(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
            logger.info(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

        return X_train, X_test, feature_names, vectorizer

    def confusion_plot(y, pred, target_names, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=target_names, ax=ax, colorbar=False)
        return fig

    def train_and_eval(X_train, X_test, y_train, y_test, feature_names, target_names, clf):
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        results = classification_report(y_test, pred, target_names=target_names, digits=4, output_dict=True)
        new_df = pd.DataFrame({f"{task}={k}": v for k, v in results.items()}).T.reset_index()
        if (config.output_dir / "results.csv").exists():
            df = pd.read_csv(config.output_dir / "results.csv")
            new_df = pd.concat([df, new_df])
        new_df.to_csv(config.output_dir / "results.csv", index=False)

        conf_fig = confusion_plot(y_test, pred, target_names_simplified, figsize=(5, 5))
        plt.xticks(rotation=45, ha="right")
        plt.title("Confusion matrix")
        plt.tight_layout()
        conf_fig.savefig(config.output_dir / f"plots/clf_confusion_matrix_{task}.pdf")
        conf_fig.show()

        return pred

    X_train, X_test, feature_names, vectorizer = load_dataset(data_train, data_test, verbose=True)
    joblib.dump(vectorizer, config.output_dir / "vectorizer.pkl", compress=True)

    classifiers = {}
    for task in config.data.tasks.keys():
        logger.info(task)
        y_train, target_names = load_labels(config, data_train, task)
        y_test, target_names = load_labels(config, data_test, task)
        target_names_simplified = [tn.split("=")[1] for tn in target_names]
        clf = RidgeClassifier(alpha=0.5, tol=1e-2, solver="sparse_cg", class_weight="balanced")

        train_and_eval(
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names,
            target_names_simplified,
            clf,
        )
        classifiers[task] = clf
        feat_fig, predictive_words = plot_feature_effects(X_train, clf, feature_names, target_names_simplified)
        logger.info("top 5 keywords per class:")
        top_words = predictive_words[:5]
        logger.info(top_words)
        
        feat_fig.suptitle("Average feature effect")
        plt.savefig(config.output_dir / f"plots/clf_features_{task}.pdf")
        plt.show()

        with open(config.output_dir / f"{task}.pkl", "wb") as f:
            dump(clf, f, protocol=5)
