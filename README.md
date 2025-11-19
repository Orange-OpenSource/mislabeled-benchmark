# Mislabeled exemples calibration detection benchmark

**Calibration improves detection of mislabeled examples**  *Ilies Chibane, Thomas George, Pierre Nodet, Vincent Lemaire* https://arxiv.org/pdf/2511.02738

EN. This repository serves to reproduce the benchmark and figures of the paper. It mainly requires the library 'mislabeled' available at https://github.com/Orange-OpenSource/mislabeled.

FR. Ce dépôt sert à reproduire les expériences et figures de l'article. Il se base principalement sur la bibliothèque 'mislabeled' disponible sur https://github.com/Orange-OpenSource/mislabeled.

## Experiments for calibration set noisy
parallel uv run python3 benchmark_detect.py --corruption weak --mode calibration --dataset yoruba hausa census bank-marketing trec professor_teacher tennis yelp bioresponse agnews imdb basketball amazon commercial mushroom phishing sms spambase youtube --output ecai-sigmoid-2 --common-seed --calibration isotonic --calibration {} ::: isotonic sigmoid none

## Experiments for calibration set noisy
uv run python3 benchmark_detect.py --corruption weak --mode calibration --dataset yoruba hausa census bank-marketing trec professor_teacher tennis yelp bioresponse agnews imdb basketball amazon commercial mushroom phishing sms spambase youtube --output ecai-sigmoid-2 --common-seed --calibration isotonic --calibration_set noisy

## Experiments for calibration set size
parallel uv run python3 benchmark_detect.py --corruption weak --mode calibration --dataset yoruba hausa census bank-marketing trec professor_teacher tennis yelp bioresponse agnews imdb basketball amazon commercial mushroom phishing sms spambase youtube --output ecai-sigmoid-2 --calibration_set clean --common-seed --calibration isotonic --calibration_size {} ::: 0.05 0.1 0.25 0.5   

## Train classifiers from trust scores
parallel uv run python3 benchmark_estim.py --corruption weak --classifier klm --dataset {} --output ecai-output-sigmoid-2 --ts_path ecai-sigmoid-2 --common-seed --n_sampling_estim 12 ::: youtube spambase sms mushroom phishing yoruba hausa census bank-marketing trec professor_teacher tennis yelp bioresponse agnews imdb basketball amazon commercial