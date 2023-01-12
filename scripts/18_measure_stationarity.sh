#!/bin/sh
#
#SBATCH --job-name=Stationarity
#SBATCH -c 2 
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0-14

source ~/.bashrc
source activate hcnn
python 18_measure_stationarity.py $SLURM_ARRAY_TASK_ID