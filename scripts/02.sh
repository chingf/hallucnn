#!/bin/sh
#
#SBATCH --job-name=HypPnet
#SBATCH -c 3
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0-14

source ~/.bashrc
source activate a40
python 02_train_hyperparameters.py $SLURM_ARRAY_TASK_ID pnet 1960 pnet

