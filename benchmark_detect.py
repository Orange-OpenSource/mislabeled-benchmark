import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime

import h5py
import numpy as np
import scipy.sparse as sp
from autocommit import autocommit
from datasets import get_weak_datasets
from define_models import (
    detectors_adjusted,
    detectors_gb,
    detectors_klm,
    detectors_calibrated,
    kernels,
)
from mislabeled.ensemble.calibration import CalibratedEnsemble
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, PredefinedSplit

parser = argparse.ArgumentParser(prog="Mislabeled exemples detection benchmark")
parser.add_argument("--corruption", choices=["weak", "noise"], required=True)
parser.add_argument(
    "--mode", choices=["klm", "gb", "calibration", "adjust"], required=True
)
parser.add_argument("--dataset", action="store", nargs="+", required=True)
parser.add_argument(
    "--datasets_folder", default=os.path.join(os.path.expanduser("~"), "datasets")
)
parser.add_argument("--output", default="./output")

parser.add_argument("--restart_from", default="")

## Calibration specific arguments
parser.add_argument("--calibration_set", default="clean", choices=["clean", "noisy"])
parser.add_argument("--calibration_size", default=100, type=int)
parser.add_argument(
    "--calibration",
    default="isotonic",
    choices=["isotonic", "sigmoid", "temperature", "none"],
)

## Seeding arguments
# parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--common-seed", action="store_true")

args = parser.parse_args()


commit_hash = autocommit()
print(f"I saved the working directory as (possibly detached) commit {commit_hash}")


seed = 1
last_save = time.time()

## SUPPRESS WARNINGS OF CONVERGENCE FOR SGD

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


## PREPROCESSING OF WRENCH, WALN, AND WEASEL DATASETS

weak_datasets = get_weak_datasets(
    cache_folder=args.datasets_folder,
    corruption=args.corruption,
    seed=seed,
    datasets=args.dataset,
    calibration=args.mode == "calibration",
    calibration_size=args.calibration_size,
)


if args.mode == "klm":
    detectors = detectors_klm
elif args.mode == "gb":
    detectors = detectors_gb
elif args.mode == "calibration":
    detectors = detectors_calibrated
elif args.mode == "adjust":
    detectors = detectors_adjusted
else:
    raise ValueError(f"unrecognized benchmark mode : {args.mode}")

os.makedirs(args.output, exist_ok=True)

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

    if args.mode == "calibration":
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

    if args.mode == "calibration" and args.calibration != "none":
        y_calib = np.array(y_calib)
        if args.calibration_set == "noisy":
            unlabeled_calib = y_noisy_calib == -1
            y_calib_labeled = y_noisy_calib[~unlabeled_calib]

        elif args.calibration_set == "clean":
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

    coverage = 1 - np.mean(unlabeled)

    unlabeled_val = y_noisy_val == -1

    clean = y_noisy_train == y_train
    noise_ratio = 1 - np.mean(clean[~unlabeled])

    print(dataset_name, X_train.shape, X_val.shape, X_test.shape)

    labels = dataset["train"]["target_names"]
    n_classes = len(labels)

    for detector_name, detector_base, param_grid_detector in detectors:
        if args.mode == "calibration" and args.calibration != "none":
            # Calibration naming
            detector_name = detector_name + "_" + args.calibration

            if args.calibration_set == "noisy":
                detector_name += "_noisy"

            detector_name += "_" + str(args.calibration_size)

        # TODO: CLEAN (sadge)
        if "kernel" in detector_base.base_model.get_params():
            kernel, param_grid_kernel = kernels[dataset["kernel"]]
            detector_base.base_model.set_params(kernel=kernel)

        final_output_dir = os.path.join(args.output, args.corruption, detector_name)
        os.makedirs(final_output_dir, exist_ok=True)

        f = h5py.File(os.path.join(final_output_dir, f"{dataset_name}.hdf5"), "w")
        ts_store = f.create_group("trust_scores")

        if args.restart_from != "":
            previous_json_path = os.path.join(
                args.restart_from,
                args.corruption,
                detector_name,
                f"{dataset_name}.json",
            )
            previous_hdf5_path = os.path.join(
                args.restart_from,
                args.corruption,
                detector_name,
                f"{dataset_name}.hdf5",
            )
            try:
                with open(previous_json_path) as previous_json:
                    results = json.load(previous_json)
                with open(
                    os.path.join(final_output_dir, f"{dataset_name}.json"), mode="w"
                ) as output_file:
                    json.dump(results, output_file)

                for i in range(len(results)):
                    with h5py.File(previous_hdf5_path, mode="r") as previous_hdf5:
                        previous_hdf5.copy(
                            previous_hdf5[f"trust_scores/{i}"], f["trust_scores"]
                        )
            except:  # noqa: E722
                results = []
        else:
            results = []

        hyperparams_seed = seed if args.common_seed else None

        for params_i, params in enumerate(
            ParameterSampler(param_grid_detector, 12, random_state=hyperparams_seed)
        ):
            if params_i < len(results):
                print("skipped", dataset_name, detector_name, params_i)
                # already performed... go to next
                continue

            start = time.perf_counter()

            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(
                f"{timestamp}: {detector_name} detecting trust scores with {params}",
                flush=True,
            )

            detector = clone(detector_base).set_params(**params)

            if args.mode == "calibration" and args.calibration != "none":
                detector.ensemble = CalibratedEnsemble(
                    detector.ensemble,
                    calibration=args.calibration,
                    cv=PredefinedSplit(calibration_split),
                )

            trust_scores = detector.trust_score(X_train_labeled, y_train_labeled)
            trust_scores = np.nan_to_num(trust_scores)

            ranking_quality_noisy = np.full(n_classes, np.nan)
            ranking_quality_gt = np.full(n_classes, np.nan)
            for c in range(n_classes):
                mask_c_noisy = y_noisy_train[~unlabeled] == c
                mask_c_gt = y_train[~unlabeled] == c

                mislabeled_train_c_noisy = (
                    y_noisy_train[~unlabeled] == y_train[~unlabeled]
                )[mask_c_noisy]
                mislabeled_train_c_gt = (
                    y_noisy_train[~unlabeled] == y_train[~unlabeled]
                )[mask_c_gt]

                if len(np.unique(mislabeled_train_c_noisy)) > 1:
                    ranking_quality_noisy[c] = roc_auc_score(
                        mislabeled_train_c_noisy,
                        trust_scores[mask_c_noisy],
                    )
                if len(np.unique(mislabeled_train_c_gt)) > 1:
                    ranking_quality_gt[c] = roc_auc_score(
                        mislabeled_train_c_gt,
                        trust_scores[mask_c_gt],
                    )

            global_ranking_quality = roc_auc_score(
                y_noisy_train[~unlabeled] == y_train[~unlabeled],
                trust_scores,
            )

            end = time.perf_counter()

            res = {
                "dataset_name": dataset_name,
                "noise_ratio": noise_ratio,
                "coverage": coverage,
                "noisy_class_distribution": (
                    np.bincount(y_noisy_train[~unlabeled], minlength=n_classes)
                    / len(y_noisy_train[~unlabeled])
                ).tolist(),
                "class_distribution": (
                    np.bincount(y_train, minlength=n_classes) / len(y_train)
                ).tolist(),
                "ranking_quality_noisy": np.around(ranking_quality_noisy, 4).tolist(),
                "ranking_quality_gt": np.around(ranking_quality_gt, 4).tolist(),
                "global_ranking_quality": round(global_ranking_quality, 4),
                "detector_name": detector_name,
                "fitting_time": end - start,
                "params": params,
                "commit": commit_hash,
                "hostname": subprocess.check_output(["hostname"])
                .decode("ascii")
                .strip(),
            }

            results.append(res)

            dset = ts_store.create_dataset(
                str(params_i), shape=trust_scores.shape, dtype=trust_scores.dtype
            )
            dset[...] = trust_scores

            if time.time() - last_save > 10 or params_i == 11:
                # avoids writing on the HD too frequently
                last_save = time.time()
                f.flush()
                with open(
                    os.path.join(final_output_dir, f"{dataset_name}.json"), mode="w"
                ) as output_file:
                    json.dump(results, output_file)

        f.close()
        print({k: v for k, v in results[0].items() if k not in ["trust_scores"]})
