#!/bin/sh
#
#SBATCH --job-name=Net-Activ
#SBATCH -c 2 
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0-4

source ~/.bashrc
source activate hcnn
python 07_save_network_activations.py $SLURM_ARRAY_TASK_ID

