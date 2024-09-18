# Software Name : mislabeled-benchmark
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled-benchmark/blob/master/LICENSE.md

import numpy as np
from bqlearn.corruptions import make_label_noise
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mislabeled.datasets.cifar_n import fetch_cifar_n
from mislabeled.datasets.weasel import fetch_weasel
from mislabeled.datasets.west_african_languages import fetch_west_african_language_news
from mislabeled.datasets.wrench import fetch_wrench
from mislabeled.preprocessing import WeakLabelEncoder


def ohe_bioresponse(X, n_categories=100):
    n_features = X.shape[1]
    to_ohe = []
    for i in range(n_features):
        if len(np.unique(X[:, i])) < n_categories:
            to_ohe.append(i)
    return to_ohe


cpu_datasets = (
    (
        "bank-marketing",
        fetch_wrench,
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                [1, 2, 3, 8, 9, 10, 15],
            ),
            remainder=StandardScaler(),
        ),
        "rbf",
    ),
    (
        "bioresponse",
        fetch_wrench,
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", dtype=np.float32), ohe_bioresponse),
            remainder=StandardScaler(),
        ),
        "rbf",
    ),
    ("census", fetch_wrench, StandardScaler(), "rbf"),
    (
        "mushroom",
        fetch_wrench,
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                [0, 1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20],
            ),
            remainder=StandardScaler(),
        ),
        "rbf",
    ),
    (
        "phishing",
        fetch_wrench,
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                [1, 6, 7, 13, 14, 15, 25, 28],
            ),
            remainder=StandardScaler(),
        ),
        "rbf",
    ),
    ("spambase", fetch_wrench, StandardScaler(), "rbf"),
    (
        "sms",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
        ),
        "linear",
    ),
    (
        "youtube",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
        ),
        "linear",
    ),
    (
        "yoruba",
        fetch_west_african_language_news,
        TfidfVectorizer(strip_accents="unicode", min_df=5, max_df=0.5),
        "linear",
    ),
    (
        "hausa",
        fetch_west_african_language_news,
        TfidfVectorizer(strip_accents="unicode", min_df=5, max_df=0.5),
        "linear",
    ),
)

gpu_datasets = (
    (
        "agnews",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    ("basketball", fetch_wrench, StandardScaler(), "rbf"),
    ("commercial", fetch_wrench, StandardScaler(), "rbf"),
    (
        "imdb",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    ("tennis", fetch_wrench, StandardScaler(), "rbf"),
    (
        "trec",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=5, max_df=0.5
        ),
        "linear",
    ),
    (
        "yelp",
        fetch_wrench,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    (
        "imdb136",
        fetch_weasel,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    (
        "amazon",
        fetch_weasel,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    (
        "professor_teacher",
        fetch_weasel,
        TfidfVectorizer(
            strip_accents="unicode", stop_words="english", min_df=1e-3, max_df=0.5
        ),
        "linear",
    ),
    ("cifar10", fetch_cifar_n, StandardScaler(), "rbf"),
)

all_datasets = cpu_datasets + gpu_datasets

datasets_ranked_by_time = [
    "youtube",
    "spambase",
    "sms",
    "mushroom",
    "phishing",
    "yoruba",
    "hausa",
    "census",
    "bank-marketing",
    "trec",
    "professor_teacher",
    "tennis",
    "yelp",
    "bioresponse",
    "agnews",
    "imdb",
    "imdb136",
    "basketball",
    "amazon",
    "commercial",
    "cifar10",
]

all_datasets = sorted(all_datasets, key=lambda x: datasets_ranked_by_time.index(x[0]))


def get_weak_datasets(
    cache_folder, corruption, datasets=datasets_ranked_by_time, seed=1
):
    weak_datasets = {}
    for name, fetch, preprocessing, kernel in all_datasets:
        if name not in datasets:
            continue
        weak_dataset = {
            split: fetch(name, split=split, cache_folder=cache_folder)
            for split in ["train", "test"]
        }
        # if exists, use validation set
        try:
            weak_dataset["validation"] = fetch(
                name, split="validation", cache_folder=cache_folder
            )
        # otherwise split test set in two
        except:
            weak_dataset["validation"] = {}
            (
                weak_dataset["validation"]["data"],
                weak_dataset["test"]["data"],
                weak_dataset["validation"]["target"],
                weak_dataset["test"]["target"],
                weak_dataset["validation"]["weak_targets"],
                weak_dataset["test"]["weak_targets"],
            ) = train_test_split(
                weak_dataset["test"]["data"],
                weak_dataset["test"]["target"],
                weak_dataset["test"]["weak_targets"],
                train_size=0.2,
                random_state=seed,
                stratify=weak_dataset["test"]["target"],
            )

        if corruption == "weak":
            weak_targets = [
                weak_dataset[split]["weak_targets"]
                for split in ["train", "validation", "test"]
            ]
            weak_targets = np.concatenate(weak_targets)
            wle = WeakLabelEncoder(random_state=seed).fit(weak_targets)
            soft_wle = WeakLabelEncoder(random_state=seed, method="soft").fit(
                weak_targets
            )

            for split in ["train", "validation", "test"]:
                weak_dataset[split]["noisy_target"] = wle.transform(
                    weak_dataset[split]["weak_targets"]
                )
                weak_dataset[split]["soft_targets"] = soft_wle.transform(
                    weak_dataset[split]["weak_targets"]
                )

        elif corruption == "noise":
            for split in ["train", "validation", "test"]:
                weak_dataset[split]["noisy_target"] = make_label_noise(
                    weak_dataset[split]["target"],
                    "uniform",
                    noise_ratio=0.3,
                    random_state=seed,
                )
                weak_dataset[split]["soft_targets"] = None
        else:
            raise ValueError(f"Unknown corruption : {corruption}")

        if preprocessing is not None:
            preprocessing.fit(weak_dataset["train"]["data"])
            for split in ["train", "validation", "test"]:
                weak_dataset[split]["raw"] = weak_dataset[split]["data"]
                weak_dataset[split]["data"] = preprocessing.transform(
                    weak_dataset[split]["raw"]
                )
        weak_dataset["kernel"] = kernel
        weak_datasets[name] = weak_dataset

    return weak_datasets
