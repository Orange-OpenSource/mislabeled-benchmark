# Software Name : mislabeled-benchmark
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled-benchmark/blob/master/LICENSE.md

import os

import pandas as pd


def load_estim(result_dir, discard_datasets=[], discard_methods=[]):
    results = []
    methods = os.listdir(result_dir)
    methods = list(filter(lambda method: method not in discard_methods, methods))

    for method in methods:
        for fname in os.listdir(os.path.join(result_dir, method)):
            dataset, ext = fname.split(".")
            if ext != "json" or dataset in discard_datasets:
                continue

            with open(os.path.join(result_dir, method, f"{dataset}.json")) as f:
                results.append(pd.read_json(f, orient="records"))

    return pd.concat(results).reset_index(drop=True)
