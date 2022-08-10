#!/bin/sh
#
#SBATCH --job-name=PNet-Hyperparams 
#SBATCH -c 2 
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0-10

source ~/.bashrc
source activate hcnn
python 03_train_hyperparameters.py $SLURM_ARRAY_TASK_ID

