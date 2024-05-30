import argparse
import sys
import time
import warnings
from sklearn import clone
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from autocommit import autocommit
from datasets import get_weak_datasets
from define_models import klm, param_grid_klm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
import numpy as np
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog="Tuning min_df")
parser.add_argument(
    "--dataset",
    action="store",
    nargs="+",
    default=[
        "youtube",
        "sms",
        "yoruba",
        "hausa",
        "agnews",
        "imdb",
        "trec",
        "yelp",
        "imdb136",
        "amazon",
        "professor_teacher",
    ],
)
parser.add_argument(
    "--datasets_folder", default=os.path.join(os.path.expanduser("~"), "datasets")
)
parser.add_argument("--n_runs", default=10)
parser.add_argument("--output", default="./output")
args = parser.parse_args()


commit_hash = autocommit()
print(f"I saved the working directory as (possibly detached) commit {commit_hash}")


seed = 1
last_save = time.time()

## SUPPRESS WARNINGS OF CONVERGENCE FOR SGD

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


seed = 1
weak_datasets = get_weak_datasets(
    cache_folder=args.datasets_folder,
    corruption="weak",
    seed=seed,
    datasets=args.dataset,
)

for dataset_name, dataset in weak_datasets.items():
    (
        raw_train,
        raw_val,
        raw_test,
        y_train,
        y_val,
        y_test,
    ) = (
        dataset["train"]["raw"],
        dataset["validation"]["raw"],
        dataset["test"]["raw"],
        dataset["train"]["target"],
        dataset["validation"]["target"],
        dataset["test"]["target"],
    )

    print(dataset_name)

    raw_all = raw_train + raw_val + raw_test
    classes = np.unique(np.concatenate((y_train, y_val, y_test)))

    all_scores = []
    vocab_sizes = []
    min_dfs = range(1, 100)
    for run in range(args.n_runs):
        print(f"run {run}")
        clf = clone(klm)
        clf.set_params(sgd__random_state=seed + run)
        scores = []
        for min_df in min_dfs:
            tfidf = TfidfVectorizer(
                strip_accents="unicode", stop_words="english", min_df=min_df
            ).fit(raw_all)

            if run == 0:
                vocab_sizes.append(len(tfidf.vocabulary_.keys()))

            X_train = tfidf.transform(raw_train).astype(np.float32)
            X_val = tfidf.transform(raw_val).astype(np.float32)
            X_test = tfidf.transform(raw_test).astype(np.float32)

            split = np.concatenate(
                (-np.ones(X_train.shape[0]), np.zeros(X_val.shape[0]))
            )
            X = sp.vstack((X_train, X_val))
            y = np.concatenate((y_train, y_val))
            cv = PredefinedSplit(split)

            rs = RandomizedSearchCV(
                clf,
                param_distributions=param_grid_klm,
                n_iter=24,
                n_jobs=-1,
                cv=cv,
                scoring="neg_log_loss",
            )
            rs.fit(X, y)
            scores.append(log_loss(y_test, rs.predict_proba(X_test), labels=classes))
    all_scores.append(scores)

    plt.plot(min_dfs, np.mean(all_scores, axis=0))
    plt.fill_between(
        min_dfs,
        np.mean(all_scores, axis=0) + np.std(all_scores, axis=0),
        np.mean(all_scores, axis=0) - np.std(all_scores, axis=0),
        alpha=0.3,
    )
    output = os.path.join(args.output, "min_df")
    os.makedirs(output, exist_ok=True)
    plt.savefig(os.path.join(output, f"{dataset_name}_loss.pdf"))
    plt.clf()
    plt.close()
    plt.plot(min_dfs, vocab_sizes)
    plt.savefig(os.path.join(output, f"{dataset_name}_vocab.pdf"))
    plt.clf()
    plt.close()
    argmin = np.argmin(np.mean(all_scores, axis=0))
    print(min_dfs[argmin], vocab_sizes[argmin])
