#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 07_calc_train_utterance_prototype.py pnet 0
python 09_calc_factorization.py pnet 0 0
