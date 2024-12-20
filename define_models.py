## KERNELS DEFINITIONS

import os
from functools import partial

import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import loguniform, uniform
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV, check_cv
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
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
    InfluenceDetector,
    LinearVoSG,
    RepresenterDetector,
    SmallLoss,
    TracIn,
    VoSG,
)
from mislabeled.ensemble import (
    AbstractEnsemble,
    IndependentEnsemble,
    LeaveOneOutEnsemble,
    NoEnsemble,
    ProgressiveEnsemble,
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
from mislabeled.probe._linear import linearize_linear_model
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


@staged_fit.register(CalibratedClassifierCV)
def staged_fit_cccv(calibrator, X, y):
    """
    Perform staged fitting for CalibratedClassifierCV.

    Parameters
    ----------
    calibrator : CalibratedClassifierCV
        The calibrated classifier to fit in stages.
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.

    Yields
    ------
    cloned : CalibratedClassifierCV
        A calibrated classifier fitted on a subset of the data.
    """
    estimator = calibrator.estimator
    cv = check_cv(calibrator.cv, y=y, classifier=is_classifier(estimator))
    train, test = next(cv.split(X, y, groups=None))
    stages = staged_fit(estimator, X[train, :], y[train])
    for stage in stages:
        cloned = clone(calibrator)
        cloned.set_params(cv="prefit", estimator=stage)
        yield cloned.fit(X[test], y[test])


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
    return linearize_linear_model(linear.best_estimator_, leaves, y)


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

agra = ModelProbingDetector(klm, NoEnsemble(), GradSimilarity(), "sum")
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
    # ("klm_aum", klm_aum, param_grid_klm_aum),
    # ("klm_forget", klm_forget, param_grid_klm_forget),
    # ("klm_cleanlab", klm_cleanlab, param_grid_klm_cleanlab),
    ("klm_consensus", klm_consensus, param_grid_klm_consensus),
    # ("influence", influence, param_grid_influence),
    # ("klm_representer", klm_representer, param_grid_representer),
    # ("tracin", tracin, param_grid_tracin),
    # ("klm_vosg", klm_vosg, param_grid_klm_vosg),
    # ("agra", agra, param_grid_agra),
    ("klm_smallloss", klm_small_loss, param_grid_klm_small_loss),
]

detectors_gb = [
    # ("gb_aum", gb_aum, param_grid_gb_aum),
    # ("gb_forget", gb_forget, param_grid_gb_forget),
    # ("sgb_forget", sgb_forget, param_grid_sgb_forget),
    # ("gb_cleanlab", gb_cleanlab, param_grid_gb_cleanlab),
    # ("gb_consensus", gb_consensus, param_grid_gb_consensus),
    # ("gb_vosg", gb_vosg, param_grid_gb_vosg),
    # ("gb_smallloss", gb_small_loss, param_grid_gb_small_loss),
]

param_grid_lin_gb = {
    "iterations": [100],
    "learning_rate": loguniform(1e-5, 1e-1),
    "reg_lambda": uniform(0, 100),
}

##LINEARIZED GB

lin_gb_vosg = LinearVoSG(gb, steps=10)
lin_gb_tracin = TracIn(gb, steps=10)
lin_gb_agra = ModelProbingDetector(gb, NoEnsemble(), GradSimilarity(), "sum")

lin_gb_influence = InfluenceDetector(gb)
lin_gb_representer = RepresenterDetector(gb)

detectors_linearized_gb = [
    ("lin_gb_vosg", lin_gb_vosg, prefix_param_grid_detector(param_grid_lin_gb)),
    ("lin_gb_tracin", lin_gb_tracin, prefix_param_grid_detector(param_grid_lin_gb)),
    ("lin_gb_agra", lin_gb_agra, prefix_param_grid_detector(param_grid_lin_gb)),
    (
        "lin_gb_influence",
        lin_gb_influence,
        prefix_param_grid_detector(param_grid_lin_gb),
    ),
    (
        "lin_gb_representer",
        lin_gb_representer,
        prefix_param_grid_detector(param_grid_lin_gb),
    ),
]

##CALIBRATION

klm_aumcal = AreaUnderMargin(
    CalibratedClassifierCV(klm, method="isotonic", ensemble=False)
)
param_grid_klm_aumcal = prefix_param_grid_detector(
    param_grid_prefix("estimator", param_grid_klm)
)
gb_aumcal = AreaUnderMargin(
    CalibratedClassifierCV(gb, method="isotonic", ensemble=False)
)
param_grid_gb_aumcal = prefix_param_grid_detector(
    param_grid_prefix("estimator", param_grid_gb)
)

klm_forget_cal = ForgetScores(
    CalibratedClassifierCV(klm, method="isotonic", ensemble=False)
)
param_grid_klm_forget_cal = prefix_param_grid_detector(
    param_grid_prefix("estimator", param_grid_klm)
)


class IndependentCalibratedEnsemble(AbstractEnsemble):
    """Ensemble of bagged models.

    Parameters
    ----------
    in_the_bag : bool, default=False
        whether to also compute probe on in_the_bag examples
    """

    def __init__(
        self,
        ensemble_strategy,
        *,
        n_jobs=None,
    ):
        self.ensemble_strategy = ensemble_strategy
        self.n_jobs = n_jobs

    def probe_model(self, calibrator, X, y, probe):

        n_samples = X.shape[0]

        def no_scoring(estimator, X, y):
            return 0

        estimator = calibrator.estimator
        cv = check_cv(calibrator.cv, y=y, classifier=is_classifier(estimator))
        train, test = next(cv.split(X, y, groups=None))

        results = cross_validate(
            estimator,
            X[train, :],
            y[train],
            cv=self.ensemble_strategy,
            n_jobs=self.n_jobs,
            scoring=no_scoring,
            return_indices=True,
            return_estimator=True,
        )

        members = []
        for member in results["estimator"]:
            cloned = clone(calibrator)
            cloned.set_params(cv="prefit", estimator=member)
            members.append(cloned.fit(X[test], y[test]))

        probe_scores = (probe(member, X, y) for member in members)

        oobs = []
        for indices_oob in results["indices"]["test"]:
            oob = np.zeros(n_samples, dtype=bool)
            oob[indices_oob] = True
            oobs.append(oob)

        return probe_scores, dict(oobs=oobs)


klm_cleanlabcal = ModelProbingDetector(
    CalibratedClassifierCV(klm, method="isotonic", ensemble=False),
    IndependentCalibratedEnsemble(
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=1,
            random_state=seed,
        ),
        n_jobs=-1,
    ),
    probe="confidence",
    aggregate=oob(mean),
)
param_grid_klm_cleanlabcal = prefix_param_grid_detector(
    param_grid_prefix("estimator", param_grid_klm)
)

klm_consensus_calibrated = ModelProbingDetector(
    CalibratedClassifierCV(klm, method="isotonic", ensemble=False),
    IndependentCalibratedEnsemble(
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=1,
            random_state=seed,
        ),
        n_jobs=-1,
    ),
    probe="accuracy",
    aggregate=oob(mean),
)
param_grid_klm_consensus_calibrated = prefix_param_grid_detector(
    param_grid_prefix("estimator", param_grid_klm)
)


class NoEnsembleCalibrated(AbstractEnsemble):
    """A no-op Ensemble"""

    def probe_model(self, calibrator, X, y, probe):

        estimator = calibrator.estimator
        cv = check_cv(calibrator.cv, y=y, classifier=is_classifier(estimator))
        train, test = next(cv.split(X, y, groups=None))

        base_model = clone(estimator)
        base_model.fit(X[train, :], y[train])
        cloned = clone(calibrator)
        cloned.set_params(cv="prefit", estimator=base_model)
        probe_scores = probe(cloned.fit(X[test], y[test]), X, y)

        return [probe_scores], {}


klm_smallloss_calibrated = ModelProbingDetector(
    CalibratedClassifierCV(klm, method="isotonic", ensemble=False),
    NoEnsembleCalibrated(),
    probe="cross_entropy",
    aggregate="sum",
)
param_grid_klm_smallloss_calibrated = prefix_param_grid_detector(
    param_grid_prefix("estimator", param_grid_klm)
)

detectors_calibrated = [
    (
        "klm_aum_calibrated",
        klm_aumcal,
        param_grid_klm_aumcal,
    ),
    # (
    #     "klm_forget_calibrated",
    #     klm_forget_cal,
    #     param_grid_klm_forget_cal,
    # ),
    # (
    #     "gb_aum_calibrated",
    #     gb_aumcal,
    #     param_grid_gb_aumcal,
    # ),
    (
        "klm_cleanlab_calibrated",
        klm_cleanlabcal,
        param_grid_klm_cleanlabcal,
    ),
    (
        "klm_consensus_calibrated",
        klm_consensus_calibrated,
        param_grid_klm_consensus_calibrated,
    ),
    (
        "klm_smallloss_calibrated",
        klm_smallloss_calibrated,
        param_grid_klm_smallloss_calibrated,
    ),
]

detectors_calibrated_noisy = [
    (f"{n}_noisy", d, g) for (n, d, g) in detectors_calibrated
]

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

klm_cleanlab_adjusted = ModelProbingDetector(
    klm,
    IndependentEnsemble(
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=1,
            random_state=seed,
        ),
        n_jobs=-1,
    ),
    probe=Confidence(Adjust(Probabilities())),
    aggregate=oob(mean),
)
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


klm_smallloss_calibrated = ModelProbingDetector(
    CalibratedClassifierCV(klm, method="isotonic", ensemble=False),
    NoEnsembleCalibrated(),
    probe=CrossEntropy(Adjust(Probabilities())),
    aggregate="sum",
)
param_grid_klm_smallloss_calibrated = prefix_param_grid_detector(
    param_grid_prefix("estimator", param_grid_klm)
)


detectors_adjusted = [
    # (
    #     "klm_aum_adjusted",
    #     klm_aum_adjusted,
    #     param_grid_adjusted_aum,
    # ),
    # (
    #     "gb_aum_adjusted",
    #     gb_aum_adjusted,
    #     param_grid_gb_adjusted_aum,
    # ),
    # (
    #     "klm_forget_adjusted",
    #     klm_forget_adjusted,
    #     param_grid_klm_forget_adjusted,
    # ),
    # (
    #     "klm_cleanlab_adjusted",
    #     klm_cleanlab_adjusted,
    #     param_grid_cleanlab_adjusted,
    # ),
    # (
    #     "klm_consensus_adjusted",
    #     klm_consensus_adjusted,
    #     param_grid_klm_consensus_adjusted,
    # ),
    # (
    #     "klm_smallloss_adjusted",
    #     klm_consensus_adjusted,
    #     param_grid_klm_consensus_adjusted,
    # ),
]


## AGRA SPECIFIC DETECTORS DEFINITION

progressive_agra = ModelProbingDetector(
    klm, ProgressiveEnsemble(), GradSimilarity(), "sum"
)
param_grid_progressive_agra = prefix_param_grid_detector(param_grid_klm)


def derivative(scores, masks):
    return scores[:, :, -1] - scores[:, :, 0]


forget_agra = ModelProbingDetector(
    klm, ProgressiveEnsemble(), GradSimilarity(), derivative
)
param_grid_forget_agra = prefix_param_grid_detector(param_grid_klm)

independent_agra = ModelProbingDetector(
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
    GradSimilarity(),
    "sum",
)
param_grid_independent_agra = prefix_param_grid_detector(param_grid_klm)

oob_agra = ModelProbingDetector(
    klm,
    IndependentEnsemble(
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=10,
            random_state=seed,
        ),
        n_jobs=-1,
    ),
    GradSimilarity(),
    "mean_oob",
)
param_grid_oob_agra = prefix_param_grid_detector(param_grid_klm)

loss = ModelProbingDetector(klm, NoEnsemble(), "entropy", "sum")
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
    detectors_knn
    + detectors_klm
    + detectors_gb
    + detectors_agra
    + detectors_baseline
    + detectors_linearized_gb
    + detectors_calibrated
    + detectors_calibrated_noisy
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
