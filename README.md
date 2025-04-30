# Mislabeled exemples calibration detection benchmark

 uv run python3 benchmark_detect.py --corruption weak --mode calibration --dataset yoruba hausa census bank-marketing trec professor_teacher tennis yelp bioresponse agnews imdb basketball amazon commercial mushroom phishing sms spambase youtube --output ecai-sigmoid --calibration isotonic --calibration_set noisy --calibration_size 0.2

parallel uv run python3 benchmark_estim.py --corruption weak --classifier klm --dataset {} --output ecai-output-sigmoid --ts_path ecai-sigmoid ::: youtube spambase sms mushroom phishing yoruba hausa census bank-marketing trec professor_teacher tennis yelp bioresponse agnews imdb basketball amazon commercial