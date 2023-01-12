#!/bin/sh
#
#SBATCH --job-name=Hyp_SNR-9
#SBATCH -c 2 
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0-14

source ~/.bashrc
source activate hcnn
python 04_train_hyperparameters.py $SLURM_ARRAY_TASK_ID pnet_snr-9 7060 pnet_snr-9_pt2 6
