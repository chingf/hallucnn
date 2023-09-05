#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

### pnet_freq_shuffle

python 05_save_validation_activations.py 0 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 0 &
python 05_save_validation_activations.py 1 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 1 &
python 05_save_validation_activations.py 2 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 2 &
python 05_save_validation_activations.py 3 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 3 &
wait
python 05_save_validation_activations.py 4 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 0 &
python 05_save_validation_activations.py 5 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 1 &
python 05_save_validation_activations.py 6 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 2 &
python 05_save_validation_activations.py 7 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 3 &
wait
python 05_save_validation_activations.py 8 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 0 &
python 05_save_validation_activations.py 9 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 1 &
python 05_save_validation_activations.py 10 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 2 &
python 05_save_validation_activations.py 11 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 3 &
wait
python 05_save_validation_activations.py 12 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 0 &
python 05_save_validation_activations.py 13 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 1 &
python 05_save_validation_activations.py 14 pnet_freq_shuffle pnet_freq_shuffle 2300 pnet_freq_shuffle 2 &
wait


### pnet_freq_shuffle 2

python 05_save_validation_activations.py 0 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 0 &
python 05_save_validation_activations.py 1 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 1 &
python 05_save_validation_activations.py 2 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 2 &
python 05_save_validation_activations.py 3 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 3 &
wait
python 05_save_validation_activations.py 4 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 0 &
python 05_save_validation_activations.py 5 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 1 &
python 05_save_validation_activations.py 6 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 2 &
python 05_save_validation_activations.py 7 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 3 &
wait
python 05_save_validation_activations.py 8 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 0 &
python 05_save_validation_activations.py 9 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 1 &
python 05_save_validation_activations.py 10 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 2 &
python 05_save_validation_activations.py 11 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 3 &
wait
python 05_save_validation_activations.py 12 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 0 &
python 05_save_validation_activations.py 13 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 1 &
python 05_save_validation_activations.py 14 pnet_freq_shuffle2 pnet_freq_shuffle2 1750 pnet_freq_shuffle2 2 &
wait


### pnet_freq_shuffle 3

python 05_save_validation_activations.py 0 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 0 &
python 05_save_validation_activations.py 1 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 1 &
python 05_save_validation_activations.py 2 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 2 &
python 05_save_validation_activations.py 3 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 3 &
wait
python 05_save_validation_activations.py 4 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 0 &
python 05_save_validation_activations.py 5 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 1 &
python 05_save_validation_activations.py 6 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 2 &
python 05_save_validation_activations.py 7 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 3 &
wait
python 05_save_validation_activations.py 8 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 0 &
python 05_save_validation_activations.py 9 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 1 &
python 05_save_validation_activations.py 10 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 2 &
python 05_save_validation_activations.py 11 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 3 &
wait
python 05_save_validation_activations.py 12 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 0 &
python 05_save_validation_activations.py 13 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 1 &
python 05_save_validation_activations.py 14 pnet_freq_shuffle3 pnet_freq_shuffle3 2000 pnet_freq_shuffle3 2 &
wait


### PNET-NOISY

python 05_save_validation_activations.py 0 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 0 &
python 05_save_validation_activations.py 1 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 1 &
python 05_save_validation_activations.py 2 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 2 &
python 05_save_validation_activations.py 3 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 3 &
wait
python 05_save_validation_activations.py 4 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 0 &
python 05_save_validation_activations.py 5 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 1 &
python 05_save_validation_activations.py 6 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 2 &
python 05_save_validation_activations.py 7 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 3 &
wait
python 05_save_validation_activations.py 8 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 0 &
python 05_save_validation_activations.py 9 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 1 &
python 05_save_validation_activations.py 10 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 2 &
python 05_save_validation_activations.py 11 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 3 &
wait
python 05_save_validation_activations.py 12 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 0 &
python 05_save_validation_activations.py 13 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 1 &
python 05_save_validation_activations.py 14 pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle 2 &
wait


### PNET-NOISY2

python 05_save_validation_activations.py 0 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 0 &
python 05_save_validation_activations.py 1 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 1 &
python 05_save_validation_activations.py 2 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 2 &
python 05_save_validation_activations.py 3 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 3 &
wait
python 05_save_validation_activations.py 4 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 0 &
python 05_save_validation_activations.py 5 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 1 &
python 05_save_validation_activations.py 6 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 2 &
python 05_save_validation_activations.py 7 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 3 &
wait
python 05_save_validation_activations.py 8 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 0 &
python 05_save_validation_activations.py 9 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 1 &
python 05_save_validation_activations.py 10 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 2 &
python 05_save_validation_activations.py 11 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 3 &
wait
python 05_save_validation_activations.py 12 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 0 &
python 05_save_validation_activations.py 13 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 1 &
python 05_save_validation_activations.py 14 pnet_temp_shuffle2 pnet_temp_shuffle2 2000 pnet_temp_shuffle2 2 &
wait


### PNET-NOISY3

python 05_save_validation_activations.py 0 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 0 &
python 05_save_validation_activations.py 1 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 1 &
python 05_save_validation_activations.py 2 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 2 &
python 05_save_validation_activations.py 3 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 3 &
wait
python 05_save_validation_activations.py 4 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 0 &
python 05_save_validation_activations.py 5 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 1 &
python 05_save_validation_activations.py 6 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 2 &
python 05_save_validation_activations.py 7 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 3 &
wait
python 05_save_validation_activations.py 8 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 0 &
python 05_save_validation_activations.py 9 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 1 &
python 05_save_validation_activations.py 10 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 2 &
python 05_save_validation_activations.py 11 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 3 &
wait
python 05_save_validation_activations.py 12 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 0 &
python 05_save_validation_activations.py 13 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 1 &
python 05_save_validation_activations.py 14 pnet_temp_shuffle3 pnet_temp_shuffle3 2000 pnet_temp_shuffle3 2 &
wait

