#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 11_calc_activity_norm.py pnet pnet 1960 pnet
python 11_calc_activity_norm.py pnet2 pnet2 2000 pnet2
python 11_calc_activity_norm.py pnet3 pnet3 1785 pnet3

python 11_calc_activity_norm.py pnet_noisy pnet_noisy 2000 pnet_noisy
python 11_calc_activity_norm.py pnet_noisy2 pnet_noisy2 2000 pnet_noisy2
python 11_calc_activity_norm.py pnet_noisy3 pnet_noisy3 2000 pnet_noisy3

python 11_calc_activity_norm.py pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle
python 11_calc_activity_norm.py pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2
python 11_calc_activity_norm.py pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3

python 11_calc_activity_norm.py pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle
python 11_calc_activity_norm.py pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2
python 11_calc_activity_norm.py pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3
