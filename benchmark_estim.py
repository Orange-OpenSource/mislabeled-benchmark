# Software Name : mislabeled-benchmark
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled-benchmark/blob/master/LICENSE.md

import argparse
from functools import partial
import json
import os
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime

import h5py
import numpy as np
import scipy.sparse as sp
from autocommit import autocommit
from datasets import get_weak_datasets
from define_models import (
    baselines,
    classifiers,
    detectors_all,
    param_grid_prefix,
    splitters,
    kernels,
)
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler

from mislabeled.handle import FilterClassifier
from mislabeled.split import PerClassSplitter

seed = 1

parser = argparse.ArgumentParser(prog="Mislabeled exemples detection benchmark")
parser.add_argument("--corruption", choices=["weak", "noise"], required=True)
parser.add_argument("--classifier", choices=["klm", "gb"], required=True)
parser.add_argument("--dataset", action="store", nargs="+", required=True)
parser.add_argument("--detector", action="store", nargs="+", required=True)
parser.add_argument(
    "--datasets_folder", default=os.path.join(os.path.expanduser("~"), "datasets")
)

parser.add_argument("--output", default="./output")
parser.add_argument("--ts_path", help="Folder where trust scores are stored")
parser.add_argument("--restart_from", default="")
parser.add_argument("--strategy", default="filter", choices=["filter", "relabel"])
parser.add_argument("--by_class", action="store_true")
parser.add_argument("--n_sampling_estim", type=int, default=3)

args = parser.parse_args()
commit_hash = autocommit()
print(f"I saved the working directory as (possibly detached) commit {commit_hash}")

## Not implemented

if args.strategy == "relabel" and args.by_class:
    raise NotImplementedError()

## SUPPRESS WARNINGS OF CONVERGENCE FOR SGD

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


detectors = [d for d in detectors_all if d[0] in args.detector]


def random_trust_scores(seed, size):
    ts = np.arange(size)
    np.random.default_rng(seed=seed).shuffle(ts)
    return ts


class TrustScoreReader:

    def __init__(self, base_path, dataset, detector):

        with open(os.path.join(base_path, detector, f"{dataset}.json"), mode="r") as f:
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
)
os.makedirs(args.output, exist_ok=True)

classifier, param_grid_classifier = classifiers[args.classifier]

for dataset_name, dataset in weak_datasets.items():
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
    unlabeled = y_noisy_train == -1
    X_train_labeled = X_train[~unlabeled]

    # FASTER TRAINING
    X_train_labeled = X_train_labeled.astype(np.float32)
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if sp.issparse(X_train):
        X_train_labeled = sp.csc_matrix(X_train_labeled)
        X_train = sp.csc_matrix(X_train)
        X_val = sp.csc_matrix(X_val)
        X_test = sp.csc_matrix(X_test)

    else:
        X_train_labeled = np.asfortranarray(X_train_labeled)
        X_train = np.asfortranarray(X_train)
        X_val = np.asfortranarray(X_val)
        X_test = np.asfortranarray(X_test)

    y_train = np.array(y_train)
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

    for detector_name, *_ in detectors:
        final_output_dir = os.path.join(
            args.output, args.corruption, args.classifier, detector_name
        )
        os.makedirs(final_output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        print(f"{timestamp}: handler for {dataset_name} | {detector_name}")
        if detector_name not in baselines:
            splitter, param_grid_splitter = splitters[detector_name]
            if args.by_class:
                splitter = PerClassSplitter(splitter)
            if detector_name != "random":
                try:
                    trust_score_reader = TrustScoreReader(
                        os.path.join(args.ts_path, args.corruption),
                        dataset_name,
                        detector_name,
                    )
                except:
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
                with open(previous_json_path, mode="r") as previous_json:
                    results = json.load(previous_json)
                with open(
                    os.path.join(final_output_dir, f"{dataset_name}.json"), mode="w"
                ) as output_file:
                    json.dump(results, output_file)
                to_skip = len(results)

            except:
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
                    if detector_name == "gold":
                        model.fit(X_train, y_train)
                    elif detector_name == "white_gold":
                        model.fit(X_train_labeled, y_train[~unlabeled])
                    elif detector_name == "silver":
                        model.fit(X_train[clean, :], y_noisy_train[clean])
                    elif detector_name == "wood":
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

                    y_pred_val = model.predict(X_val)
                    y_proba_val = model.predict_proba(X_val)

                    acc_val = accuracy_score(y_val, y_pred_val)
                    bacc_val = balanced_accuracy_score(y_val, y_pred_val)
                    kappa_val = cohen_kappa_score(y_val, y_pred_val)
                    logl_val = log_loss(y_val, y_proba_val)

                    acc_noisy_val = accuracy_score(
                        y_noisy_val[~unlabeled_val], y_pred_val[~unlabeled_val]
                    )
                    bacc_noisy_val = balanced_accuracy_score(
                        y_noisy_val[~unlabeled_val], y_pred_val[~unlabeled_val]
                    )
                    kappa_noisy_val = cohen_kappa_score(
                        y_noisy_val[~unlabeled_val], y_pred_val[~unlabeled_val]
                    )
                    logl_noisy_val = log_loss(
                        y_noisy_val[~unlabeled_val], y_proba_val[~unlabeled_val]
                    )

                    y_pred_test = model.predict(X_test)
                    y_proba_test = model.predict_proba(X_test)

                    acc_test = accuracy_score(y_test, y_pred_test)
                    bacc_test = balanced_accuracy_score(y_test, y_pred_test)
                    kappa_test = cohen_kappa_score(y_test, y_pred_test)
                    logl_test = log_loss(y_test, y_proba_test)

                    end = time.perf_counter()

                    stats["acc_test"] = acc_test
                    stats["bacc_test"] = bacc_test
                    stats["kappa_test"] = kappa_test
                    stats["logl_test"] = logl_test

                    stats["acc_val"] = acc_val
                    stats["bacc_val"] = bacc_val
                    stats["kappa_val"] = kappa_val
                    stats["logl_val"] = logl_val

                    stats["acc_noisy_val"] = acc_noisy_val
                    stats["bacc_noisy_val"] = bacc_noisy_val
                    stats["kappa_noisy_val"] = kappa_noisy_val
                    stats["logl_noisy_val"] = logl_noisy_val

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
