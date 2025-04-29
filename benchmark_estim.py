import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial

import h5py
import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from autocommit import autocommit
from datasets import get_weak_datasets
from define_models import baselines, classifiers, kernels, param_grid_prefix, splitters
from relplot import multiclass_logits_to_confidences, smECE
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler, PredefinedSplit

from mislabeled.handle import FilterClassifier
from mislabeled.split import PerClassSplitter

seed = 1

parser = argparse.ArgumentParser(prog="Mislabeled exemples detection benchmark")
parser.add_argument("--corruption", choices=["weak", "noise"], required=True)
parser.add_argument("--classifier", choices=["klm", "gb"], required=True)
parser.add_argument("--dataset", action="store", nargs="+", required=True)
parser.add_argument(
    "--datasets_folder", default=os.path.join(os.path.expanduser("~"), "datasets")
)
parser.add_argument("--output", default="./output")
parser.add_argument("--ts_path", help="Folder where trust scores are stored")
parser.add_argument("--restart_from", default="")
parser.add_argument("--strategy", default="filter", choices=["filter", "relabel"])
parser.add_argument("--by_class", action="store_true")
parser.add_argument("--n_sampling_estim", type=int, default=3)
parser.add_argument(
    "--calibration",
    choices=["none", "sigmoid", "isotonic", "temperature"],
    default="none",
)

args = parser.parse_args()
commit_hash = autocommit()
print(f"I saved the working directory as (possibly detached) commit {commit_hash}")

## Not implemented

if args.strategy == "relabel" and args.by_class:
    raise NotImplementedError

## SUPPRESS WARNINGS OF CONVERGENCE FOR SGD

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def random_trust_scores(seed, size):
    ts = np.arange(size)
    np.random.default_rng(seed=seed).shuffle(ts)
    return ts


class TrustScoreReader:
    def __init__(self, base_path, dataset, detector):
        with open(os.path.join(base_path, detector, f"{dataset}.json")) as f:
            self.results_json = json.load(f)
        self.results_hdf5 = h5py.File(
            os.path.join(base_path, detector, f"{dataset_name}.hdf5"), "r"
        )

        assert len(self.results_hdf5["trust_scores"]) == len(self.results_json)

    def get(self, i):
        return self.results_json[i], self.results_hdf5[f"trust_scores/{i}"][...]

    def length(self):
        return len(self.results_json)


class CachedTrustScoresDetector(BaseEstimator):
    def __init__(self, trust_scores):
        self.trust_scores = trust_scores

    def trust_score(self, X, y):
        assert X.shape[0] == self.trust_scores.shape[0]

        return self.trust_scores


if args.by_class:
    prefix_param_grid_splitter = partial(param_grid_prefix, "splitter__splitter")
else:
    prefix_param_grid_splitter = partial(param_grid_prefix, "splitter")


weak_datasets = get_weak_datasets(
    cache_folder=args.datasets_folder,
    corruption=args.corruption,
    seed=seed,
    datasets=args.dataset,
    calibration=args.calibration != "none",
    calibration_size=0.2,
)
os.makedirs(args.output, exist_ok=True)

classifier, param_grid_classifier = classifiers[args.classifier]

for dataset_name, dataset in weak_datasets:
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        y_noisy_train,
        y_noisy_val,
        y_noisy_test,
        y_soft_train,
        y_soft_val,
        y_soft_test,
    ) = (
        dataset["train"]["data"],
        dataset["validation"]["data"],
        dataset["test"]["data"],
        dataset["train"]["target"],
        dataset["validation"]["target"],
        dataset["test"]["target"],
        dataset["train"]["noisy_target"],
        dataset["validation"]["noisy_target"],
        dataset["test"]["noisy_target"],
        dataset["train"]["soft_targets"],
        dataset["validation"]["soft_targets"],
        dataset["test"]["soft_targets"],
    )

    if args.calibration != "none":
        (
            X_calib,
            y_calib,
            y_noisy_calib,
            y_soft_calib,
        ) = (
            dataset["calibration"]["data"],
            dataset["calibration"]["target"],
            dataset["calibration"]["noisy_target"],
            dataset["calibration"]["soft_targets"],
        )

    # FASTER TRAINING
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    unlabeled = y_noisy_train == -1

    if sp.issparse(X_train):
        X_train_labeled = sp.csc_matrix(X_train[~unlabeled])
        X_val = sp.csc_matrix(X_val)
        X_test = sp.csc_matrix(X_test)

    else:
        X_train_labeled = np.asfortranarray(X_train[~unlabeled])
        X_val = np.asfortranarray(X_val)
        X_test = np.asfortranarray(X_test)

    y_train = np.array(y_train)
    y_train_labeled = y_noisy_train[~unlabeled]

    if args.calibration != "none":
        y_calib = np.array(y_calib)

        unlabeled_calib = y_calib == -1
        y_calib_labeled = y_calib[~unlabeled_calib]

        y_train_labeled = np.concatenate((y_noisy_train[~unlabeled], y_calib_labeled))
        calibration_split = np.concatenate(
            (
                -np.ones(X_train[~unlabeled].shape[0]),
                np.zeros(X_calib[~unlabeled_calib].shape[0]),
            )
        )
        if sp.issparse(X_calib):
            X_train_labeled = sp.csc_matrix(
                sp.vstack((X_train[~unlabeled], X_calib[~unlabeled_calib]))
            )
        else:
            X_train_labeled = np.asfortranarray(
                np.vstack((X_train[~unlabeled], X_calib[~unlabeled_calib]))
            )

    clean = y_noisy_train == y_train

    unlabeled_val = y_noisy_val == -1
    if np.all(unlabeled_val):
        unlabeled_val[:] = False  # covers cifar10 case

    print(dataset_name, X_train.shape, X_test.shape)

    labels = dataset["train"]["target_names"]
    n_classes = len(labels)

    # TODO: CLEAN (sadge)
    if "kernel" in classifier.get_params():
        kernel, param_grid_kernel = kernels[dataset["kernel"]]
        classifier.set_params(kernel=kernel)

    detectors = os.listdir(os.path.join(args.ts_path, args.corruption))
    detectors = detectors + [(d, None) for d in ["none", "random", "silver", "gold"]]

    for detector_name in ["none"]:
        # for detector_name, *_ in detectors:
        final_output_dir = os.path.join(
            args.output, args.corruption, args.classifier, detector_name
        )
        os.makedirs(final_output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        print(f"{timestamp}: handler for {dataset_name} | {detector_name}")
        if detector_name not in baselines:
            # splitter, param_grid_splitter = splitters[detector_name]
            splitter, param_grid_splitter = splitters[
                "_".join(detector_name.split("_")[:2])
            ]
            if args.by_class:
                splitter = PerClassSplitter(splitter)
            if detector_name != "random":
                try:
                    trust_score_reader = TrustScoreReader(
                        os.path.join(args.ts_path, args.corruption),
                        dataset_name,
                        detector_name,
                    )
                except:  # noqa: E722
                    print("skipped (reading hdf5 likely failed)")
                    continue

        to_skip = 0
        if args.restart_from != "":
            previous_json_path = os.path.join(
                args.restart_from,
                args.corruption,
                args.classifier,
                detector_name,
                f"{dataset_name}.json",
            )
            try:
                with open(previous_json_path) as previous_json:
                    results = json.load(previous_json)
                with open(
                    os.path.join(final_output_dir, f"{dataset_name}.json"), mode="w"
                ) as output_file:
                    json.dump(results, output_file)
                to_skip = len(results)

            except:  # noqa: E722
                print(f"I could not restart from specified path {previous_json_path}")
                results = []
        else:
            results = []

        skipped = 0
        for params_i, params_classifier in enumerate(
            ParameterSampler(param_grid_classifier, 12 * args.n_sampling_estim)
        ):
            if detector_name in baselines:
                model = clone(classifier)
                model.set_params(**params_classifier)

                splitter_grid = [{}]
            else:
                if detector_name == "random":
                    trust_scores = random_trust_scores(seed=1, size=np.sum(~unlabeled))
                else:
                    stats_detector, trust_scores = trust_score_reader.get(
                        params_i % trust_score_reader.length()
                    )
                detector = CachedTrustScoresDetector(trust_scores)

                classifier_ = clone(classifier)
                classifier_.set_params(**params_classifier)

                if args.calibration != "none":
                    classifier_ = CalibratedClassifierCV(
                        classifier_,
                        ensemble=False,
                        cv=PredefinedSplit(calibration_split),
                        method=args.calibration,
                    )
                    detector_name = detector_name + "_" + args.calibration

                if args.strategy == "filter":
                    model = FilterClassifier(
                        detector,
                        splitter,
                        classifier_,
                    )

                    splitter_grid = ParameterGrid(param_grid_splitter)
                elif args.strategy == "relabel":
                    model = classifier_
                    splitter_grid = [{}]

                    indices_to_relabel = np.argsort(trust_scores)[
                        : round(0.1 * len(trust_scores))
                    ]
                    y_relabeled = deepcopy(y_noisy_train[~unlabeled])
                    assert len(trust_scores) == len(y_relabeled)
                    y_relabeled[indices_to_relabel] = y_train[~unlabeled][
                        indices_to_relabel
                    ]

            for params_splitter in splitter_grid:
                start = time.perf_counter()

                if skipped < to_skip:
                    print("skipped", dataset_name, detector_name, params_i)
                    # already performed... go to next
                    skipped += 1
                    continue

                if detector_name in baselines or detector_name == "random":
                    stats = {
                        "detector_name": detector_name,
                        "dataset_name": dataset_name,
                    }
                else:
                    stats = dict(stats_detector)
                    stats["params_detector"] = stats.pop("params")

                    if args.by_class:
                        stats["by_class"] = True

                try:
                    if detector_name.startswith("gold"):
                        model.fit(X_train, y_train)
                    elif detector_name.startswith("white_gold"):
                        model.fit(X_train_labeled, y_train[~unlabeled])
                    elif detector_name.startswith("silver"):
                        model.fit(X_train[clean, :], y_noisy_train[clean])
                    elif detector_name.startswith("none"):
                        model.fit(X_train_labeled, y_noisy_train[~unlabeled])
                    elif detector_name.startswith("wood"):
                        rng = np.random.RandomState(seed)
                        y_wood_train = y_noisy_train.copy()
                        y_wood_train[unlabeled] = rng.choice(
                            n_classes - 1, size=np.sum(unlabeled)
                        )
                        model.fit(X_train, y_wood_train)
                    elif args.strategy == "relabel":
                        model.set_params(**prefix_param_grid_splitter(params_splitter))
                        model.fit(X_train_labeled, y_relabeled)
                        stats["strategy"] = "relabel"
                    else:
                        model.set_params(**prefix_param_grid_splitter(params_splitter))
                        model.fit(X_train_labeled, y_noisy_train[~unlabeled])
                        stats["strategy"] = "filter"

                    y_pred_train = model.predict(X_train)
                    y_pred_val = model.predict(X_val)

                    y_proba_train = model.predict_proba(X_train)
                    y_proba_val = model.predict_proba(X_val)

                    y_pred_test = model.predict(X_test)
                    y_proba_test = model.predict_proba(X_test)

                    # top-label calibration metrics
                    for split, y_proba, y_pred, y in zip(
                        ["train", "noisy_val", "val", "test"],
                        [y_proba_train[~unlabeled], y_proba_val[~unlabeled_val], y_proba_val, y_proba_test],
                        [y_pred_train[~unlabeled], y_pred_val[~unlabeled_val], y_pred_val, y_pred_test],
                        [y_noisy_train[~unlabeled], y_noisy_val[~unlabeled_val], y_val, y_test],
                    ):
                        y_proba_max, agreement = multiclass_logits_to_confidences(
                            y_proba, y, probs=True
                        )
                        ece = smECE(f=y_proba_max, y=agreement)
                        stats[f"ece_{split}"] = ece

                        stats[f"acc_{split}"] = accuracy_score(y, y_pred)
                        stats[f"bacc_{split}"] = balanced_accuracy_score(y, y_pred)
                        stats[f"kappa_{split}"] = cohen_kappa_score(y, y_pred)
                        stats[f"logl_{split}"] = log_loss(y, y_proba)

                    end = time.perf_counter()

                except Exception as e:
                    import traceback

                    print(e)
                    print(traceback.print_exc())
                    end = time.perf_counter()

                stats["params_classifier"] = params_classifier
                stats["classifier_name"] = args.classifier
                stats["params_splitter"] = params_splitter

                stats["estim_time"] = end - start
                stats["hostname_estim"] = (
                    subprocess.check_output(["hostname"]).decode("ascii").strip()
                )
                stats["commit_estim"] = commit_hash

                results.append(stats)
                # print(stats)

                with open(
                    os.path.join(final_output_dir, f"{dataset_name}.json"), mode="w"
                ) as output_file:
                    json.dump(results, output_file)

            print(results[-1])
