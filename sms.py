# Software Name : mislabeled-benchmark
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled-benchmark/blob/master/LICENSE.md

# %%
import os

import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
)

from mislabeled.datasets.wrench import fetch_wrench
from mislabeled.preprocessing import WeakLabelEncoder

# %%
dataset_name = "sms"
dataset = fetch_wrench(
    dataset_name, cache_folder=os.path.join(os.path.expanduser("~"), "datasets")
)
seed = 1

# %%
X, y, y_weak = dataset["data"], dataset["target"], dataset["weak_targets"]
y_noisy = WeakLabelEncoder(random_state=seed).fit_transform(y_weak)
# y_noisy[y_noisy == -1] = 0
print(y, LabelEncoder().fit_transform(y))
y = LabelEncoder().fit_transform(y)

# targets = {k: dataset["target_names"][k] for k in range(len(dataset["target_names"]))}
# targets[-1] = "Unlabeled"
# print(targets)

# adult = fetch_openml(data_id=1590, as_frame=True)
# targets = adult["target"]
# features = adult["data"]
# features["us-native"] = (features["native-country"] == "United-States").astype(
#     "category"
# )
# features = features.drop(["education-num"], axis=1)
# features = features.drop(["native-country"], axis=1)

# noisy_targets = shuffle(targets.copy(), random_state=1).reset_index(drop=True)
# X = features
# y = LabelEncoder().fit_transform(targets)
# y_noisy = LabelEncoder().fit_transform(noisy_targets)
# %%
unlabeled = y_noisy == -1
X = [d for d, u in zip(X, unlabeled) if not u]
y_noisy = y_noisy[~unlabeled]
y = y[~unlabeled]

# %%


detect_path = "detect"
import json

import h5py
import pandas as pd


class TrustScoreReader:

    def __init__(self, base_path, dataset, detector):

        with open(os.path.join(base_path, detector, f"{dataset}.json")) as f:
            self.results_json = json.load(f)
        self.results_hdf5 = h5py.File(
            os.path.join(base_path, detector, f"{dataset}.hdf5"), "r"
        )

        assert len(self.results_hdf5["trust_scores"]) == len(self.results_json)

    def get(self, i):
        return self.results_json[i], self.results_hdf5[f"trust_scores/{i}"][...]

    def length(self):
        return len(self.results_json)


# %%
ts = TrustScoreReader("detect/weak/", dataset_name, "klm_aum").get(11 - 1)[1]
print(len(X), len(ts))
calib_ts = TrustScoreReader("detect/weak", dataset_name, "klm_aum_calibrated").get(
    7 - 1
)[1]
top = 5
table = pd.DataFrame(
    zip(
        [X[i] for i in np.argsort(ts)[0:top]],
        [dataset["target_names"][y_noisy[i]] for i in np.argsort(ts)[0:top]],
        [dataset["target_names"][y[i]] for i in np.argsort(ts)[0:top]],
    ),
    columns=["Text", "Noisy Label", "True Label"],
    index=["Baseline" for _ in range(top)],
)
table2 = pd.DataFrame(
    zip(
        [X[i] for i in np.argsort(calib_ts)[0:top]],
        [dataset["target_names"][y_noisy[i]] for i in np.argsort(calib_ts)[0:top]],
        [dataset["target_names"][y[i]] for i in np.argsort(calib_ts)[0:top]],
    ),
    columns=["Text", "Noisy Label", "True Label"],
    index=["Calibrated" for _ in range(top)],
)
table = pd.concat([table, table2])
table["Text"] = table["Text"].str.slice(stop=280)
# table["Text"] = table["Text"].map(lambda s: f"{s} ...")

#%%
print(table.to_latex(escape=True, column_format="lp{0.6\\textwidth}ll", multirow=True))

# %%
print(table)
# table["Text"]= table["Text"].map(lambda s: "\parbox[t]{5cm}{" + s + "}")
# print("\n".join([X[i] for i in np.argsort(ts)[0:top]]))
# print("\n".join([f"{y_noisy[i]} {y[i]}" for i in np.argsort(ts)[0:top]]))
# print("\n".join([X[i] for i in np.argsort(calib_ts)[0:top]]))
# print("\n".join([f"{y_noisy[i]} {y[i]}" for i in np.argsort(calib_ts)[0:top]]))
# print(dataset["target_names"])
# import matplotlib.pyplot as plt


# %%
