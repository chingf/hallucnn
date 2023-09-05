#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 10_eval_reconstruction_r2.py pnet_noisy2 2000 1
python 10_eval_reconstruction_r2.py pnet_noisy3 2000 1 

python 10_eval_reconstruction_r2.py pnet_freq_shuffle3 2000 0
python 10_eval_reconstruction_r2.py pnet_temp_shuffle3 2000 0
python 10_eval_reconstruction_r2.py pnet_noisy 2000 0
python 10_eval_reconstruction_r2.py pnet_noisy2 2000 0
python 10_eval_reconstruction_r2.py pnet_noisy3 2000 0 
