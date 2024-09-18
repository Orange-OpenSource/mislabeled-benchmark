# Software Name : mislabeled-benchmark
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled-benchmark/blob/master/LICENSE.md

## KERNELS DEFINITIONS

import os
from functools import partial

from catboost import CatBoostClassifier
from scipy.stats import loguniform, uniform
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from mislabeled.detect import ModelBasedDetector
from mislabeled.detect.detectors import (
    AreaUnderMargin,
    ConfidentLearning,
    ConsensusConsistency,
    ForgetScores,
    InfluenceDetector,
    LinearVoSG,
    RepresenterDetector,
    SmallLoss,
    TracIn,
    VoSG,
)
from mislabeled.ensemble import (
    IndependentEnsemble,
    LeaveOneOutEnsemble,
    NoEnsemble,
    ProgressiveEnsemble,
)
from mislabeled.ensemble._progressive import staged_fit
from mislabeled.probe import LinearGradSimilarity
from mislabeled.split import QuantileSplitter, ThresholdSplitter

seed = 1

rbf = RBFSampler(gamma="scale", n_components=1000, random_state=seed)

kernels = {}
kernels["rbf"] = (rbf, {})
kernels["linear"] = ("passthrough", {})

gpu_device = os.getenv("GPU_DEVICE_ORDINAL", "")
if gpu_device != "":
    print(f"I am using GPU #{gpu_device}")
else:
    print("no GPU found")


## DEFINITION FOR PROGRESSIVE ENSEMBLE
@staged_fit.register(CatBoostClassifier)
def staged_fit_cat(estimator: CatBoostClassifier, X, y):
    estimator.fit(X, y)
    for i in range(estimator.tree_count_):
        shrinked = estimator.copy()
        shrinked.shrink(i + 1)
        yield shrinked


# BASE MODEL DEFINITIONS

knn = KNeighborsClassifier()
param_grid_knn = {"n_neighbors": [1, 3, 5], "metric": ["euclidean"]}

gb = CatBoostClassifier(
    early_stopping_rounds=5,
    eval_fraction=0.1,
    verbose=0,
    random_state=seed,
    thread_count=-1,
    task_type="GPU",
    # devices=gpu_device,
    max_bin=32,
    boosting_type="Plain",
    allow_writing_files=False,
)
param_grid_gb = {
    "learning_rate": loguniform(1e-5, 1e-1),
    "reg_lambda": uniform(0, 100),
}

klm = Pipeline(
    [
        ("kernel", None),
        (
            "sgd",
            SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ],
)
param_grid_klm = {
    "sgd__alpha": loguniform(1e-5, 10**-2.5),
    "sgd__eta0": loguniform(1e-3, 1e0),
}


def param_grid_prefix(prefix, param_grid_in):
    param_grid_out = {}
    for k, v in param_grid_in.items():
        param_grid_out[f"{prefix}__" + k] = v
    return param_grid_out


prefix_param_grid_detector = partial(param_grid_prefix, "base_model")
prefix_param_grid_splitter = partial(param_grid_prefix, "splitter")


classifiers = {
    "klm": (klm, param_grid_klm),
    "gb": (gb, param_grid_gb),
}

## DETECTORS DEFINITION

knn_loo = ModelBasedDetector(knn, LeaveOneOutEnsemble(n_jobs=-1), "accuracy", "sum")
param_grid_knn_loo = prefix_param_grid_detector(param_grid_knn)

gb_aum = AreaUnderMargin(gb)
param_grid_gb_aum = prefix_param_grid_detector(param_grid_gb)

sgb_aum = AreaUnderMargin(gb)
param_grid_sgb_aum = prefix_param_grid_detector(param_grid_gb)
param_grid_sgb_aum["base_model__subsample"] = [0.33]
param_grid_sgb_aum["base_model__bootstrap_type"] = ["Poisson"]

klm_aum = AreaUnderMargin(klm)
param_grid_klm_aum = prefix_param_grid_detector(param_grid_klm)

gb_forget = ForgetScores(gb)
param_grid_gb_forget = prefix_param_grid_detector(param_grid_gb)

sgb_forget = ForgetScores(gb)
param_grid_sgb_forget = prefix_param_grid_detector(param_grid_gb)
param_grid_sgb_forget["base_model__subsample"] = [0.05]
# param_grid_sgb_forget["base_model__iterations"] = [4000]
param_grid_sgb_forget["base_model__bootstrap_type"] = ["Bernoulli"]

klm_forget = ForgetScores(klm)
param_grid_klm_forget = prefix_param_grid_detector(param_grid_klm)

# Set confident n_repeats to 1 as in cleanlab
gb_cleanlab = ConfidentLearning(gb, n_repeats=1, random_state=seed)
param_grid_gb_cleanlab = prefix_param_grid_detector(param_grid_gb)

klm_cleanlab = ConfidentLearning(klm, n_repeats=1, n_jobs=-1, random_state=seed)
param_grid_klm_cleanlab = prefix_param_grid_detector(param_grid_klm)

gb_consensus = ConsensusConsistency(gb, random_state=seed)
param_grid_gb_consensus = prefix_param_grid_detector(param_grid_gb)

klm_consensus = ConsensusConsistency(klm, n_jobs=-1, random_state=seed)
param_grid_klm_consensus = prefix_param_grid_detector(param_grid_klm)

influence = InfluenceDetector(klm)
param_grid_influence = prefix_param_grid_detector(param_grid_klm)

klm_representer = RepresenterDetector(klm)
param_grid_representer = prefix_param_grid_detector(param_grid_klm)

tracin = TracIn(klm)
param_grid_tracin = prefix_param_grid_detector(param_grid_klm)

gb_vosg = VoSG(gb, n_directions=100, steps=5, random_state=seed)
param_grid_gb_vosg = prefix_param_grid_detector(param_grid_gb)

klm_vosg = LinearVoSG(klm)
param_grid_klm_vosg = prefix_param_grid_detector(param_grid_klm)

agra = ModelBasedDetector(klm, NoEnsemble(), LinearGradSimilarity(), "sum")
param_grid_klm_agra = param_grid_klm.copy()
param_grid_klm_agra["sgd__fit_intercept"] = [True, False]
param_grid_agra = prefix_param_grid_detector(param_grid_klm_agra)

gb_small_loss = SmallLoss(gb)
param_grid_gb_small_loss = prefix_param_grid_detector(param_grid_gb)

klm_small_loss = SmallLoss(klm)
param_grid_klm_small_loss = prefix_param_grid_detector(param_grid_klm)


detectors_knn = [
    # ("knn_loo", knn_loo, param_grid_knn_loo),
]

detectors_klm = [
    ("klm_aum", klm_aum, param_grid_klm_aum),
    ("klm_forget", klm_forget, param_grid_klm_forget),
    ("klm_cleanlab", klm_cleanlab, param_grid_klm_cleanlab),
    ("klm_consensus", klm_consensus, param_grid_klm_consensus),
    ("influence", influence, param_grid_influence),
    ("klm_representer", klm_representer, param_grid_representer),
    ("tracin", tracin, param_grid_tracin),
    ("klm_vosg", klm_vosg, param_grid_klm_vosg),
    ("agra", agra, param_grid_agra),
    ("klm_smallloss", klm_small_loss, param_grid_klm_small_loss),
]

detectors_gb = [
    ("gb_aum", gb_aum, param_grid_gb_aum),
    ("gb_forget", gb_forget, param_grid_gb_forget),
    ("sgb_forget", sgb_forget, param_grid_sgb_forget),
    ("gb_cleanlab", gb_cleanlab, param_grid_gb_cleanlab),
    ("gb_consensus", gb_consensus, param_grid_gb_consensus),
    ("gb_vosg", gb_vosg, param_grid_gb_vosg),
    ("gb_smallloss", gb_small_loss, param_grid_gb_small_loss),
]

## AGRA SPECIFIC DETECTORS DEFINITION

progressive_agra = ModelBasedDetector(
    klm, ProgressiveEnsemble(), LinearGradSimilarity(), "sum"
)
param_grid_progressive_agra = prefix_param_grid_detector(param_grid_klm)


def derivative(scores, masks):
    return scores[:, :, -1] - scores[:, :, 0]


forget_agra = ModelBasedDetector(
    klm, ProgressiveEnsemble(), LinearGradSimilarity(), derivative
)
param_grid_forget_agra = prefix_param_grid_detector(param_grid_klm)

independent_agra = ModelBasedDetector(
    klm,
    IndependentEnsemble(
        StratifiedShuffleSplit(
            train_size=0.7,
            n_splits=50,
            random_state=seed,
        ),
        n_jobs=-1,
        # in_the_bag=True,
    ),
    LinearGradSimilarity(),
    "sum",
)
param_grid_independent_agra = prefix_param_grid_detector(param_grid_klm)

oob_agra = ModelBasedDetector(
    klm,
    IndependentEnsemble(
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=10,
            random_state=seed,
        ),
        n_jobs=-1,
    ),
    LinearGradSimilarity(),
    "mean_oob",
)
param_grid_oob_agra = prefix_param_grid_detector(param_grid_klm)

loss = ModelBasedDetector(klm, NoEnsemble(), "entropy", "sum")
param_grid_loss = prefix_param_grid_detector(param_grid_klm)

detectors_agra = [
    ("agra", agra, param_grid_agra),
    ("progressive_agra", progressive_agra, param_grid_progressive_agra),
    ("independent_agra", independent_agra, param_grid_independent_agra),
    ("oob_agra", oob_agra, param_grid_oob_agra),
    ("loss", loss, param_grid_loss),
]

detectors_baseline = [
    ("gold", None, None),
    ("white_gold", None, None),
    ("silver", None, None),
    ("none", None, None),
    ("wood", None, None),
    ("random", None, None),
]

detectors_all = (
    detectors_knn + detectors_klm + detectors_gb + detectors_agra + detectors_baseline
)

baselines = ["gold", "white_gold", "silver", "wood", "none"]
baseline_split = ["random"]

## SPLITTER DEFINITION

splitters = {}

quantile_splitter = QuantileSplitter()
param_grid_quantile_splitter = {
    "quantile": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}


for detector_name, *_ in detectors_all:
    if detector_name not in baselines:
        splitters[detector_name] = (quantile_splitter, param_grid_quantile_splitter)

# splitters["agra"] = (ThresholdSplitter(0), {})
splitters["progressive_agra"] = (ThresholdSplitter(0), {})
splitters["independent_agra"] = (ThresholdSplitter(0), {})
splitters["oob_agra"] = (ThresholdSplitter(0), {})

splitters["knn_loo"] = (ThresholdSplitter(1), {})
splitters["gb_consensus"] = (
    ThresholdSplitter(),
    {"threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
)
splitters["klm_consensus"] = (
    ThresholdSplitter(),
    {"threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
)
