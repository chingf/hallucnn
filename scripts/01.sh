#!/bin/sh
#
#SBATCH --job-name=Pnet2
#SBATCH -c 2 
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=0

source ~/.bashrc
source activate a40
python 01_train_net.py pnet2 2000
