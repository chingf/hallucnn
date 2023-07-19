#!/bin/sh
#
#SBATCH --job-name=Activ
#SBATCH -c 4
#SBATCH --time=99:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chingfang17@gmail.com
#SBATCH --array=8-14

source ~/.bashrc
hostname=$HOSTNAME 
echo "$hostname"
if [[ "$hostname" == "ax11" || $hostname == "ax12" ]]; then
    conda activate a40
elif [[ "$hostname" == "ax13" || "$hostname" == "ax14" ]]; then
    conda activate a40
elif [[ "$hostname" == "ax15" || "$hostname" == "ax16" ]]; then
    conda activate a40
else
    conda activate hcnn
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cf2794/.conda/envs/hcnn/lib
    export LD_LIBRARY_PATH=/home/cf2794/.conda/envs/hcnn/lib/python3.7/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
fi

python 05_save_validation_activations.py $SLURM_ARRAY_TASK_ID pnet_temp_shuffle pnet_temp_shuffle 2140 pnet_temp_shuffle

