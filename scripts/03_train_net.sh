#!/bin/sh
#
#SBATCH --job-name=CgramShuffle
#SBATCH -c 2 
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0

source ~/.bashrc
source activate hcnn
python 03_train_net_cgram_shuffle.py pnet_freq_shuffle_pt2 pnet_freq_shuffle 1180 1000
