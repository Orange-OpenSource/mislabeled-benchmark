# %%

import json
import os

import h5py
import numpy as np

from datasets import datasets_ranked_by_time

# %%

base_path = os.path.join(os.path.expanduser("~"), f"output/detect/noise")

# %%
detectors = []
datasets = []

for detector in os.listdir(base_path):
    detectors.append(detector)
    for filename in os.listdir(os.path.join(base_path, detector)):
        dataset, ext = filename.split(".")
        if ext != "json":
            continue

        datasets.append(dataset)

        h5py_path = os.path.join(base_path, detector, f"{dataset}.hdf5")
        json_path = os.path.join(base_path, detector, f"{dataset}.json")
        if not os.path.isfile(h5py_path):
            print("missing h5py", dataset, detector)

            with open(json_path) as f:
                d = json.load(f)

            results = []
            f = h5py.File(h5py_path, "w")
            ts_store = f.create_group("trust_scores")

            for i, res in enumerate(d):
                ts = np.array(res.pop("trust_scores"))

                dset = ts_store.create_dataset(str(i), shape=ts.shape, dtype=ts.dtype)
                dset[...] = ts

                results.append(res)

            with open(json_path, mode="w") as output_file:
                json.dump(results, output_file)



detectors = np.unique(detectors)
datasets = np.unique(datasets)

print(detectors)
print(datasets)
# %%
