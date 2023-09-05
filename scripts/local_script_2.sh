#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 07_fit_validation_PCA.py pnet
python 07_fit_validation_PCA.py pnet2
python 07_fit_validation_PCA.py pnet3

python 07_fit_validation_PCA.py pnet_noisy
python 07_fit_validation_PCA.py pnet_noisy2
python 07_fit_validation_PCA.py pnet_noisy3

#python 07_fit_train_prototype_PCA.py pnet n10000_sample0 0
#python 08_calc_invariance.py pnet n10000_sample0 0
#python 09_calc_factorization.py pnet n10000_sample0 0
