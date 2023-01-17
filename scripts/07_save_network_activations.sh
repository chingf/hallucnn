#!/bin/sh
#
#SBATCH --job-name=ActReg
#SBATCH -c 2 
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0-14

source ~/.bashrc
source activate hcnn
python 07_save_network_activations.py $SLURM_ARRAY_TASK_ID pnet pnet 1960 pnet_pt3
python 07_save_network_activations.py $SLURM_ARRAY_TASK_ID pnet_cgram_shuffle pnet_cgram_shuffle 2140 pnet_cgram_shuffle
python 07_save_network_activations.py $SLURM_ARRAY_TASK_ID pnet_snr-9 pnet_snr-9 7060 pnet_snr-9_pt2

