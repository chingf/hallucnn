#!/bin/sh
#
#SBATCH --job-name=Denoising
#SBATCH -c 2 
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0-14

source ~/.bashrc
source activate hcnn
python 11_calc_popln_denoising.py $SLURM_ARRAY_TASK_ID pnet
python 11_calc_popln_denoising.py $SLURM_ARRAY_TASK_ID pnet_cgram_shuffle
