#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 08_calc_invariance.py pnet 1
python 08_calc_invariance.py pnet 0 0
python 08_calc_invariance.py pnet 1 0
