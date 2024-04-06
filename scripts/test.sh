cd ./src
python inference_run.py --data_path "../data/steam" --log_path "log-steam/1" --checkpoint_path "log/1" --test_prefix "steamtest"
python inference_run.py --data_path "../data/steam" --log_path "log-steam/2" --checkpoint_path "log/2" --test_prefix "steamtest"
python inference_run.py --data_path "../data/steam" --log_path "log-steam/3" --checkpoint_path "log/3" --test_prefix "steamtest"
python inference_run.py --data_path "../data/steam" --log_path "log-steam/4" --checkpoint_path "log/4" --test_prefix "steamtest"
python inference_run.py --data_path "../data/steam" --log_path "log-steam/5" --checkpoint_path "log/5" --test_prefix "steamtest"
python avg.py
