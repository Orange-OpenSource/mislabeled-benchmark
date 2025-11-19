# Software Name : mislabeled-benchmark
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled-benchmark/blob/master/LICENSE.md

# %%
from copy import deepcopy
import os

import numpy as np
from sklearn import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from mislabeled.datasets.moons import make_moons, moons_ground_truth_pyx
from mislabeled.detect.detectors import AreaUnderMargin
from mislabeled.ensemble.calibration import CalibratedEnsemble
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, PredefinedSplit
from mislabeled.probe import Margin, Probabilities
import matplotlib.colors as mcolors
import texfig as tf
from texfig import TMLR_textwidth

# %%
X, y = make_moons(1000, spread=0.1, shuffle=True, class_imbalance=0.11, random_state=1)
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 1000),
    np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 1000),
)
XX = np.stack((xx.ravel(), yy.ravel()), axis=-1)
YY = moons_ground_truth_pyx(XX, spread=0.1, class_imbalance=0.1)
X_train, X_test, y_train, y_test, train, test = train_test_split(
    X, y, np.arange(X.shape[0]), random_state=2, stratify=y, train_size=100
)
mislabeled = np.zeros(X_train.shape[0], dtype=bool)
rng = np.random.RandomState(2)
mislabeled_maj = rng.choice(np.flatnonzero(y_train == 0), 5, replace=False)
mislabeled[mislabeled_maj] = True
mislabeled_min = rng.choice(np.flatnonzero(y_train == 1), 5, replace=False)
mislabeled[mislabeled_min] = True
y_train[mislabeled] = 1 - y_train[mislabeled]
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
split = np.zeros(X.shape[0])
split[np.arange(X_train.shape[0])] = -1
# %%
klm = make_pipeline(
    StandardScaler(),
    RBFSampler(gamma=2, n_components=200, random_state=2),
    MLPClassifier(
        hidden_layer_sizes=[],
        activation="tanh",
        solver="sgd",
        alpha=1e-2,
        max_iter=1000,
        learning_rate_init=0.01,
        n_iter_no_change=1000,
        tol=1e-16,
        batch_size=24,
        random_state=2,
    ),
    # SGDClassifier(
    #     loss="log_loss",
    #     learning_rate="constant",
    #     # early_stopping=True,
    #     # validation_fraction=0.1,
    #     # n_iter_no_change=5,
    #     tol=1e-12,
    #     random_state=2,
    #     n_jobs=-1,
    #     eta0=0.2,
    #     alpha=1e-6,
    #     max_iter=1000,
    # ),
)
aum = AreaUnderMargin(klm)
aum.probe = Margin(Probabilities())
ts_baseline = aum.trust_score(X_train, y_train)
aum_calibrated = AreaUnderMargin(klm)
aum_calibrated.ensemble = CalibratedEnsemble(
    aum_calibrated.ensemble, calibration="isotonic", cv=PredefinedSplit(split)
)
ts_calibrated = aum_calibrated.trust_score(X, y)
# %%
model = clone(klm).fit(X_train, y_train)
model_calibrated = CalibratedClassifierCV(
    deepcopy(model),
    method="isotonic",
    cv="prefit",
).fit(X_test, y_test)
# %%
plt.plot(model[-1].loss_curve_)
# %%
for i, (n, m, ts) in enumerate(
    zip(
        ["baseline", "calibrated"],
        [model, model_calibrated],
        [ts_baseline, ts_calibrated],
    )
):
    tf.figure(width=TMLR_textwidth * 2 / 5, ratio=1, pad=0.5)

    plt.contour(
        xx, yy, YY.reshape(xx.shape), levels=[0.5], colors="black", linestyles="dashed"
    )

    top = np.zeros(X_train.shape[0], dtype=bool)
    top[np.argsort(ts)[:10]] = True
    bottom = ~top

    colors = np.array(
        [
            mcolors.XKCD_COLORS["xkcd:neon purple"],
            mcolors.XKCD_COLORS["xkcd:light neon green"],
        ]
    )

    plt.scatter(
        X_train[bottom & ~mislabeled, 0],
        X_train[bottom & ~mislabeled, 1],
        c=colors[y_train[bottom & ~mislabeled]],
        s=20,
        edgecolors="black",
        linewidths=1,
        alpha=0.8,
    )
    plt.scatter(
        X_train[bottom & mislabeled, 0],
        X_train[bottom & mislabeled, 1],
        c=colors[y_train[bottom & mislabeled]],
        s=40,
        edgecolors="black",
        linewidths=1,
        marker="*",
        alpha=0.8,
    )
    plt.scatter(
        X_train[top & ~mislabeled, 0],
        X_train[top & ~mislabeled, 1],
        c=colors[y_train[top & ~mislabeled]],
        s=160,
        edgecolors="red",
        linewidths=1,
    )
    plt.scatter(
        X_train[top & mislabeled, 0],
        X_train[top & mislabeled, 1],
        c=colors[y_train[top & mislabeled]],
        s=240,
        edgecolors="red",
        linewidths=1,
        marker="*",
    )
    if i == 0:
        plt.scatter([],[], s=40, edgecolors="red", linewidths=1, facecolors="white", label="untrust.")
        plt.scatter([],[], s=40, edgecolors="black", linewidths=1, facecolors="black", marker="*", label="mislab.")
        plt.scatter([],[], s=20, edgecolors="black", linewidths=1, facecolors="black", label="clean.")
        plt.legend(loc="upper right", edgecolor="black", handletextpad=0.4, handlelength=1.5)
    plt.xticks(())
    plt.yticks(())
    # ax.scatter([], [], s=60, marker="*", color="black", label="mislabeled")
    # ax.scatter([], [], s=20, color="black", label="clean")
    # plt.legend()
    # plt.tight_layout()
    # plt.xlabel(n, fontweight="bold")
    plt.text(
        -0.4,
        -0.65,
        n,
        fontdict={"weight": "bold", "font": "Times New Roman", "size": 12},
        bbox=dict(facecolor='none', edgecolor='black', linewidth=1, boxstyle='round,pad=0.2')
    )
    os.makedirs("figures", exist_ok=True)
    tf.savefig(f"figures/two-moons-{i}")
# %%
