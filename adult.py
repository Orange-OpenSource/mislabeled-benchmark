# %%
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV, check_cv
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import PredefinedSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from mislabeled.detect.detectors import AreaUnderMargin
from mislabeled.ensemble import staged_fit
from mislabeled.datasets.wrench import fetch_wrench
from mislabeled.datasets.weasel import fetch_weasel
from mislabeled.preprocessing import WeakLabelEncoder
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

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
import numpy as np
import pandas as pd
from scipy.stats import entropy


class TrustScoreReader:

    def __init__(self, base_path, dataset, detector):

        with open(os.path.join(base_path, detector, f"{dataset}.json"), mode="r") as f:
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

# %%
# print(table)
# table["Text"]= table["Text"].map(lambda s: "\parbox[t]{5cm}{" + s + "}")
print(table.to_latex(escape=True, column_format="lp{0.6\\textwidth}ll", multirow=True))
# print("\n".join([X[i] for i in np.argsort(ts)[0:top]]))
# print("\n".join([f"{y_noisy[i]} {y[i]}" for i in np.argsort(ts)[0:top]]))
# print("\n".join([X[i] for i in np.argsort(calib_ts)[0:top]]))
# print("\n".join([f"{y_noisy[i]} {y[i]}" for i in np.argsort(calib_ts)[0:top]]))
# print(dataset["target_names"])
# calib_ts_in = calibrated_ts[split == -1]
# ts_in = ts[split == -1]
# y_in = y[split == -1]
# X_in = X[split == -1]
# n_classes = len(np.unique(y))
# clean_per_row = 3
# baseline_per_row = 2
# calibrated_per_row = 2
# tot_per_row = clean_per_row + baseline_per_row + calibrated_per_row

# fig, ax = plt.subplots(n_classes, 6, figsize=(10, 9))

# fig.suptitle("Top self representer values of an MLP on MNIST")

# for xx, c in enumerate(np.unique(y)):
#     slice = y_in == c
#     big_diff = np.argsort(ts_in[slice]) - np.argsort(calib_ts_in[slice])
#     for yy, top_i in enumerate(np.argsort(-big_diff)[0:3]):
#         ax[xx, yy].imshow(X_in[slice][top_i].reshape(28, 28))
#         ax[xx, yy].set_xticks(())
#         ax[xx, yy].set_yticks(())
#     for yy, top_i in enumerate(np.argsort(big_diff)[0:3]):
#         yy += 3
#         ax[xx, yy].imshow(X_in[slice][top_i].reshape(28, 28))
#         ax[xx, yy].set_xticks(())
#         ax[xx, yy].set_yticks(())
# slice = y_in == c
# for yy, top_i in enumerate(np.argsort(ts_in[slice])[0:baseline_per_row]):
#     ax[xx, yy].imshow(X_in[slice][top_i].reshape(28, 28))
#     ax[xx, yy].set_xticks(())
#     ax[xx, yy].set_yticks(())
# for yy, top_i in enumerate(np.argsort(calib_ts_in[slice])[0:calibrated_per_row]):
#     # xx, yy = i // img_per_row, i % img_per_row
#     yy += baseline_per_row
#     ax[xx, yy].imshow(X_in[slice][top_i].reshape(28, 28))
#     # ax[xx, yy].set_title(f"label : {y[top_i]}")
#     ax[xx, yy].set_xticks(())
#     ax[xx, yy].set_yticks(())
# for yy, top_i in enumerate(np.argsort(-calib_ts_in[slice])[0:clean_per_row]):
#     # xx, yy = i // img_per_row, i % img_per_row
#     yy += baseline_per_row + calibrated_per_row
#     ax[xx, yy].imshow(X_in[slice][top_i].reshape(28, 28))
#     # ax[xx, yy].set_title(f"label : {y[top_i]}")
#     ax[xx, yy].set_xticks(())
#     ax[xx, yy].set_yticks(())
# plt.tight_layout()
# plt.show()

# %%
