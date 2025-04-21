# Mislabeled exemples detection benchmark

python3 benchmark_detect.py --corruption weak --mode calibration --dataset youtube spambase sms mushroom phishing yoruba hausa census bank-marketing trec professor_teacher tennis yelp bioresponse agnews imdb basketball amazon commercial --output ijcnn2 --calibration_size 0.2

python3 benchmark_estim.py --corruption weak --classifier klm --dataset youtube spambase sms mushroom phishing yoruba hausa census bank-marketing trec professor_teacher tennis yelp bioresponse agnews imdb basketball amazon commercial --output ijcnn-output --ts_path ijcnn