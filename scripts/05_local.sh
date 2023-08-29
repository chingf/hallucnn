#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

### PNET

python 05_save_validation_activations.py 0 pnet pnet 1960 pnet 0 &
python 05_save_validation_activations.py 1 pnet pnet 1960 pnet 1 &
python 05_save_validation_activations.py 2 pnet pnet 1960 pnet 2 &
python 05_save_validation_activations.py 3 pnet pnet 1960 pnet 3 &
wait
python 05_save_validation_activations.py 4 pnet pnet 1960 pnet 0 &
python 05_save_validation_activations.py 5 pnet pnet 1960 pnet 1 &
python 05_save_validation_activations.py 6 pnet pnet 1960 pnet 2 &
python 05_save_validation_activations.py 7 pnet pnet 1960 pnet 3 &
wait
python 05_save_validation_activations.py 8 pnet pnet 1960 pnet 0 &
python 05_save_validation_activations.py 9 pnet pnet 1960 pnet 1 &
python 05_save_validation_activations.py 10 pnet pnet 1960 pnet 2 &
python 05_save_validation_activations.py 11 pnet pnet 1960 pnet 3 &
wait
python 05_save_validation_activations.py 12 pnet pnet 1960 pnet 0 &
python 05_save_validation_activations.py 13 pnet pnet 1960 pnet 1 &
python 05_save_validation_activations.py 14 pnet pnet 1960 pnet 2 &
wait


### PNET 2

python 05_save_validation_activations.py 0 pnet2 pnet2 2000 pnet2 0 &
python 05_save_validation_activations.py 1 pnet2 pnet2 2000 pnet2 1 &
python 05_save_validation_activations.py 2 pnet2 pnet2 2000 pnet2 2 &
python 05_save_validation_activations.py 3 pnet2 pnet2 2000 pnet2 3 &
wait
python 05_save_validation_activations.py 4 pnet2 pnet2 2000 pnet2 0 &
python 05_save_validation_activations.py 5 pnet2 pnet2 2000 pnet2 1 &
python 05_save_validation_activations.py 6 pnet2 pnet2 2000 pnet2 2 &
python 05_save_validation_activations.py 7 pnet2 pnet2 2000 pnet2 3 &
wait
python 05_save_validation_activations.py 8 pnet2 pnet2 2000 pnet2 0 &
python 05_save_validation_activations.py 9 pnet2 pnet2 2000 pnet2 1 &
python 05_save_validation_activations.py 10 pnet2 pnet2 2000 pnet2 2 &
python 05_save_validation_activations.py 11 pnet2 pnet2 2000 pnet2 3 &
wait
python 05_save_validation_activations.py 12 pnet2 pnet2 2000 pnet2 0 &
python 05_save_validation_activations.py 13 pnet2 pnet2 2000 pnet2 1 &
python 05_save_validation_activations.py 14 pnet2 pnet2 2000 pnet2 2 &
wait


### PNET 3

python 05_save_validation_activations.py 0 pnet2 pnet2 1785 pnet2 0 &
python 05_save_validation_activations.py 1 pnet2 pnet2 1785 pnet2 1 &
python 05_save_validation_activations.py 2 pnet2 pnet2 1785 pnet2 2 &
python 05_save_validation_activations.py 3 pnet2 pnet2 1785 pnet2 3 &
wait
python 05_save_validation_activations.py 4 pnet2 pnet2 1785 pnet2 0 &
python 05_save_validation_activations.py 5 pnet2 pnet2 1785 pnet2 1 &
python 05_save_validation_activations.py 6 pnet2 pnet2 1785 pnet2 2 &
python 05_save_validation_activations.py 7 pnet2 pnet2 1785 pnet2 3 &
wait
python 05_save_validation_activations.py 8 pnet2 pnet2 1785 pnet2 0 &
python 05_save_validation_activations.py 9 pnet2 pnet2 1785 pnet2 1 &
python 05_save_validation_activations.py 10 pnet2 pnet2 1785 pnet2 2 &
python 05_save_validation_activations.py 11 pnet2 pnet2 1785 pnet2 3 &
wait
python 05_save_validation_activations.py 12 pnet2 pnet2 1785 pnet2 0 &
python 05_save_validation_activations.py 13 pnet2 pnet2 1785 pnet2 1 &
python 05_save_validation_activations.py 14 pnet2 pnet2 1785 pnet2 2 &
wait


### PNET 3

python 05_save_validation_activations.py 0 pnet3 pnet3 1785 pnet3 0 &
python 05_save_validation_activations.py 1 pnet3 pnet3 1785 pnet3 1 &
python 05_save_validation_activations.py 2 pnet3 pnet3 1785 pnet3 2 &
python 05_save_validation_activations.py 3 pnet3 pnet3 1785 pnet3 3 &
wait
python 05_save_validation_activations.py 4 pnet3 pnet3 1785 pnet3 0 &
python 05_save_validation_activations.py 5 pnet3 pnet3 1785 pnet3 1 &
python 05_save_validation_activations.py 6 pnet3 pnet3 1785 pnet3 2 &
python 05_save_validation_activations.py 7 pnet3 pnet3 1785 pnet3 3 &
wait
python 05_save_validation_activations.py 8 pnet3 pnet3 1785 pnet3 0 &
python 05_save_validation_activations.py 9 pnet3 pnet3 1785 pnet3 1 &
python 05_save_validation_activations.py 10 pnet3 pnet3 1785 pnet3 2 &
python 05_save_validation_activations.py 11 pnet3 pnet3 1785 pnet3 3 &
wait
python 05_save_validation_activations.py 12 pnet3 pnet3 1785 pnet3 0 &
python 05_save_validation_activations.py 13 pnet3 pnet3 1785 pnet3 1 &
python 05_save_validation_activations.py 14 pnet3 pnet3 1785 pnet3 2 &
wait


### PNET-NOISY

python 05_save_validation_activations.py 0 pnet_noisy pnet_noisy 2000 pnet_noisy 0 &
python 05_save_validation_activations.py 1 pnet_noisy pnet_noisy 2000 pnet_noisy 1 &
python 05_save_validation_activations.py 2 pnet_noisy pnet_noisy 2000 pnet_noisy 2 &
python 05_save_validation_activations.py 3 pnet_noisy pnet_noisy 2000 pnet_noisy 3 &
wait
python 05_save_validation_activations.py 4 pnet_noisy pnet_noisy 2000 pnet_noisy 0 &
python 05_save_validation_activations.py 5 pnet_noisy pnet_noisy 2000 pnet_noisy 1 &
python 05_save_validation_activations.py 6 pnet_noisy pnet_noisy 2000 pnet_noisy 2 &
python 05_save_validation_activations.py 7 pnet_noisy pnet_noisy 2000 pnet_noisy 3 &
wait
python 05_save_validation_activations.py 8 pnet_noisy pnet_noisy 2000 pnet_noisy 0 &
python 05_save_validation_activations.py 9 pnet_noisy pnet_noisy 2000 pnet_noisy 1 &
python 05_save_validation_activations.py 10 pnet_noisy pnet_noisy 2000 pnet_noisy 2 &
python 05_save_validation_activations.py 11 pnet_noisy pnet_noisy 2000 pnet_noisy 3 &
wait
python 05_save_validation_activations.py 12 pnet_noisy pnet_noisy 2000 pnet_noisy 0 &
python 05_save_validation_activations.py 13 pnet_noisy pnet_noisy 2000 pnet_noisy 1 &
python 05_save_validation_activations.py 14 pnet_noisy pnet_noisy 2000 pnet_noisy 2 &
wait


### PNET-NOISY2

python 05_save_validation_activations.py 0 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 0 &
python 05_save_validation_activations.py 1 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 1 &
python 05_save_validation_activations.py 2 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 2 &
python 05_save_validation_activations.py 3 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 3 &
wait
python 05_save_validation_activations.py 4 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 0 &
python 05_save_validation_activations.py 5 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 1 &
python 05_save_validation_activations.py 6 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 2 &
python 05_save_validation_activations.py 7 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 3 &
wait
python 05_save_validation_activations.py 8 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 0 &
python 05_save_validation_activations.py 9 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 1 &
python 05_save_validation_activations.py 10 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 2 &
python 05_save_validation_activations.py 11 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 3 &
wait
python 05_save_validation_activations.py 12 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 0 &
python 05_save_validation_activations.py 13 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 1 &
python 05_save_validation_activations.py 14 pnet_noisy2 pnet_noisy2 2000 pnet_noisy2 2 &
wait


### PNET-NOISY3

python 05_save_validation_activations.py 0 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 0 &
python 05_save_validation_activations.py 1 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 1 &
python 05_save_validation_activations.py 2 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 2 &
python 05_save_validation_activations.py 3 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 3 &
wait
python 05_save_validation_activations.py 4 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 0 &
python 05_save_validation_activations.py 5 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 1 &
python 05_save_validation_activations.py 6 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 2 &
python 05_save_validation_activations.py 7 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 3 &
wait
python 05_save_validation_activations.py 8 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 0 &
python 05_save_validation_activations.py 9 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 1 &
python 05_save_validation_activations.py 10 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 2 &
python 05_save_validation_activations.py 11 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 3 &
wait
python 05_save_validation_activations.py 12 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 0 &
python 05_save_validation_activations.py 13 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 1 &
python 05_save_validation_activations.py 14 pnet_noisy3 pnet_noisy3 2000 pnet_noisy3 2 &
wait

