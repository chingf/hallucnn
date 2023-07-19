#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 07_fit_train_prototype_PCA.py pnet n5000_sample0 0
python 09_calc_factorization.py pnet n5000_sample0 0

#python 05_sample_train_activations.py pnet 1 5000
#python 05_sample_train_activations.py pnet 0 10000
#
#python 07_fit_train_prototype_PCA.py pnet n5000_sample1
#python 08_calc_invariance.py pnet n5000_sample1
#python 09_calc_factorization.py pnet n5000_sample1
#
#python 07_fit_train_prototype_PCA.py pnet n10000_sample0
#python 08_calc_invariance.py pnet n10000_sample0
#python 09_calc_factorization.py pnet n10000_sample0
