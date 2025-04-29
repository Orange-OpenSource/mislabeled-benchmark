## KERNELS DEFINITIONS

import os
from functools import partial

import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import loguniform, uniform
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mislabeled.aggregate import mean, oob
from mislabeled.detect import ModelProbingDetector
from mislabeled.detect.detectors import (
    AreaUnderMargin,
    ConfidentLearning,
    ConsensusConsistency,
    ForgetScores,
    SelfInfluenceDetector,
    VoLG,
    RepresenterDetector,
    SmallLoss,
    TracIn,
    FiniteDiffVoG,
)
from mislabeled.ensemble import (
    IndependentEnsemble,
    LeaveOneOutEnsemble,
    NoEnsemble,
    staged_fit,
)
from mislabeled.probe import (
    Accuracy,
    Adjust,
    Confidence,
    CrossEntropy,
    GradSimilarity,
    Margin,
    Probabilities,
    linearize,
)
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


## DEFINITION FOR LINEAR DETECTORS
@linearize.register(CatBoostClassifier)
def linearize_catboost(estimator: CatBoostClassifier, X, y):
    leaves = OneHotEncoder().fit_transform(estimator.calc_leaf_indexes(X))
    linear = RandomizedSearchCV(
        klm[-1],
        param_distributions={
            "alpha": loguniform(1e-5, 10**-2.5),
            "eta0": loguniform(1e-3, 1e0),
        },
        n_iter=10,
        n_jobs=-1,
    )
    # linear = LogisticRegressionCV(solver="newton-cg", n_jobs=-1, max_iter=1000)
    linear.fit(leaves, y)
    return linearize(linear.best_estimator_, leaves, y)


# BASE MODEL DEFINITIONS

knn = KNeighborsClassifier()
param_grid_knn = {"n_neighbors": [1, 3, 5], "metric": ["euclidean"]}

gb = CatBoostClassifier(
    early_stopping_rounds=5,
    eval_fraction=0.1,
    verbose=0,
    random_state=seed,
    thread_count=-1,
    # task_type="GPU",
    # devices=gpu_device,
    # max_bin=32,
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

knn_loo = ModelProbingDetector(knn, LeaveOneOutEnsemble(n_jobs=-1), "accuracy", "sum")
param_grid_knn_loo = prefix_param_grid_detector(param_grid_knn)

gb_aum = AreaUnderMargin(gb, staging="predict")
gb_aum.probe = Margin(Probabilities())
param_grid_gb_aum = prefix_param_grid_detector(param_grid_gb)

sgb_aum = AreaUnderMargin(gb)
param_grid_sgb_aum = prefix_param_grid_detector(param_grid_gb)
param_grid_sgb_aum["base_model__subsample"] = [0.33]
param_grid_sgb_aum["base_model__bootstrap_type"] = ["Poisson"]

klm_aum = AreaUnderMargin(klm)
klm_aum.probe = Margin(Probabilities())
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

influence = SelfInfluenceDetector(klm)
param_grid_influence = prefix_param_grid_detector(param_grid_klm)

gb_influence = SelfInfluenceDetector(gb)

klm_representer = RepresenterDetector(klm)
param_grid_representer = prefix_param_grid_detector(param_grid_klm)

gb_representer = RepresenterDetector(gb)

tracin = TracIn(klm)
param_grid_tracin = prefix_param_grid_detector(param_grid_klm)

gb_tracin = TracIn(gb, steps=10)

gb_fd_vosg = FiniteDiffVoG(gb, n_directions=100, steps=5, random_state=seed)
param_grid_gb_fd_vosg = prefix_param_grid_detector(param_grid_gb)

klm_vosg = VoLG(klm)
param_grid_klm_vosg = prefix_param_grid_detector(param_grid_klm)

gb_vosg = VoLG(gb, steps=10)

agra = ModelProbingDetector(klm, NoEnsemble(), GradSimilarity(), "sum")
param_grid_klm_agra = param_grid_klm.copy()
param_grid_klm_agra["sgd__fit_intercept"] = [True, False]
param_grid_agra = prefix_param_grid_detector(param_grid_klm_agra)

gb_agra = ModelProbingDetector(gb, NoEnsemble(), GradSimilarity(), "sum")

gb_small_loss = SmallLoss(gb)
param_grid_gb_small_loss = prefix_param_grid_detector(param_grid_gb)

klm_small_loss = SmallLoss(klm)
param_grid_klm_small_loss = prefix_param_grid_detector(param_grid_klm)


detectors_knn = [
    ("knn_loo", knn_loo, param_grid_knn_loo),
]

detectors_klm = [
    ("klm_aum", klm_aum, param_grid_klm_aum),
    ("klm_forget", klm_forget, param_grid_klm_forget),
    ("klm_cleanlab", klm_cleanlab, param_grid_klm_cleanlab),
    ("klm_consensus", klm_consensus, param_grid_klm_consensus),
    ("klm_influence", influence, param_grid_influence),
    ("klm_representer", klm_representer, param_grid_representer),
    ("klm_tracin", tracin, param_grid_tracin),
    ("klm_vosg", klm_vosg, param_grid_klm_vosg),
    ("klm_agra", agra, param_grid_agra),
    ("klm_smallloss", klm_small_loss, param_grid_klm_small_loss),
]

detectors_gb = [
    ("gb_aum", gb_aum, param_grid_gb_aum),
    ("gb_forget", gb_forget, param_grid_gb_forget),
    ("gb_cleanlab", gb_cleanlab, param_grid_gb_cleanlab),
    ("gb_consensus", gb_consensus, param_grid_gb_consensus),
    ("gb_influence", gb_influence, prefix_param_grid_detector(param_grid_gb)),
    ("gb_representer", gb_representer, prefix_param_grid_detector(param_grid_gb)),
    ("gb_tracin", gb_tracin, prefix_param_grid_detector(param_grid_gb)),
    ("gb_vosg", gb_vosg, prefix_param_grid_detector(param_grid_gb)),
    ("gb_fd_vosg", gb_fd_vosg, param_grid_gb_fd_vosg),
    ("gb_smallloss", gb_small_loss, param_grid_gb_small_loss),
    ("gb_agra", gb_agra, prefix_param_grid_detector(param_grid_gb)),
]

## CALIBRATED

detectors_calibrated = list(
    filter(
        lambda detector: detector[0]
        in ["klm_smallloss", "klm_consensus", "klm_cleanlab", "klm_aum"],
        detectors_klm,
    )
)

# for consistency with cleanlab
for d in detectors_calibrated:
    if d[0] == "klm_consensus":
        d[1].n_repeats = 1


## ADJUSTED

klm_aum_adjusted = AreaUnderMargin(klm)
klm_aum_adjusted.probe = Margin(Adjust(Probabilities()))
param_grid_adjusted_aum = prefix_param_grid_detector(param_grid_klm)

gb_aum_adjusted = AreaUnderMargin(gb, staging="predict")
gb_aum_adjusted.probe = Margin(Adjust(Probabilities()))
param_grid_gb_adjusted_aum = prefix_param_grid_detector(param_grid_gb)

klm_forget_adjusted = ForgetScores(klm)


class ArgMax:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, model, X, y):
        return np.argmax(self.inner(model, X, y), axis=1)


klm_forget_adjusted.probe = Accuracy(ArgMax(Adjust(Probabilities())))
param_grid_klm_forget_adjusted = prefix_param_grid_detector(param_grid_klm)

klm_cleanlab_adjusted = ConfidentLearning(
    klm, n_splits=5, n_repeats=1, n_jobs=-1, random_state=seed
)
klm_cleanlab_adjusted.probe = Confidence(Adjust(Probabilities()))
param_grid_cleanlab_adjusted = prefix_param_grid_detector(param_grid_klm)

klm_consensus_adjusted = ModelProbingDetector(
    klm,
    IndependentEnsemble(
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=1,
            random_state=seed,
        ),
        n_jobs=-1,
    ),
    probe=Accuracy(ArgMax(Adjust(Probabilities()))),
    aggregate=oob(mean),
)
param_grid_klm_consensus_adjusted = prefix_param_grid_detector(param_grid_klm)


klm_smallloss_adjusted = SmallLoss(klm)
klm_smallloss_adjusted.probe = CrossEntropy(Adjust(Probabilities()))
param_grid_klm_smallloss_adjusted = prefix_param_grid_detector(param_grid_klm)


detectors_adjusted = [
    (
        "klm_aum_adjusted",
        klm_aum_adjusted,
        param_grid_adjusted_aum,
    ),
    (
        "klm_cleanlab_adjusted",
        klm_cleanlab_adjusted,
        param_grid_cleanlab_adjusted,
    ),
    (
        "klm_consensus_adjusted",
        klm_consensus_adjusted,
        param_grid_klm_consensus_adjusted,
    ),
    (
        "klm_smallloss_adjusted",
        klm_smallloss_adjusted,
        param_grid_klm_smallloss_adjusted,
    ),
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
    detectors_knn
    + detectors_klm
    + detectors_gb
    + detectors_baseline
    + detectors_adjusted
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
