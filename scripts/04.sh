#!/bin/sh
#
#SBATCH --job-name=ActBabble
#SBATCH -c 2 
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=8-13

source ~/.bashrc
source activate hcnn
python 07_save_babble_activations.py $SLURM_ARRAY_TASK_ID pnet_babble pnet 1960 pnet_pt3 0
