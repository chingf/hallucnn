#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 10_eval_reconstruction_r2.py pnet 1960 1
python 10_eval_reconstruction_r2.py pnet_freq_shuffle 2300 1
python 10_eval_reconstruction_r2.py pnet_temp_shuffle 2140 1
python 10_eval_reconstruction_r2.py pnet2 2000 1
python 10_eval_reconstruction_r2.py pnet3 1785 1
python 10_eval_reconstruction_r2.py pnet_freq_shuffle2 1750 1
python 10_eval_reconstruction_r2.py pnet_temp_shuffle2 2000 1
