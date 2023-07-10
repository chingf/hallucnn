#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 05_save_train_activations.py 0 pnet pnet 1960 pnet 0 &
python 05_save_train_activations.py 1 pnet pnet 1960 pnet 1 &
python 05_save_train_activations.py 2 pnet pnet 1960 pnet 2 &
python 05_save_train_activations.py 3 pnet pnet 1960 pnet 3 &

wait

python 05_save_train_activations.py 4 pnet pnet 1960 pnet 0 &
python 05_save_train_activations.py 5 pnet pnet 1960 pnet 1 &
python 05_save_train_activations.py 6 pnet pnet 1960 pnet 2 &
python 05_save_train_activations.py 7 pnet pnet 1960 pnet 3 &

wait

python 05_save_train_activations.py 8 pnet pnet 1960 pnet 0 &
python 05_save_train_activations.py 9 pnet pnet 1960 pnet 1 &
python 05_save_train_activations.py 10 pnet pnet 1960 pnet 2 &
python 05_save_train_activations.py 11 pnet pnet 1960 pnet 3 &

wait

python 05_save_train_activations.py 12 pnet pnet 1960 pnet 0 &
python 05_save_train_activations.py 13 pnet pnet 1960 pnet 1 &
python 05_save_train_activations.py 14 pnet pnet 1960 pnet 2 &

wait

