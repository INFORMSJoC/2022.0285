cd ../src
python kfold_split.py
python train_classifier_linear.py --data_path "../data/reddit/split/1" --log_path "log/1"
python train_classifier_linear.py --data_path "../data/reddit/split/2" --log_path "log/2"
python train_classifier_linear.py --data_path "../data/reddit/split/3" --log_path "log/3"
python train_classifier_linear.py --data_path "../data/reddit/split/4" --log_path "log/4"
python train_classifier_linear.py --data_path "../data/reddit/split/5" --log_path "log/5"
python avg.py
