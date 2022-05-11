#!/bin/bash
#SBATCH --job-name=gpt2_5000
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --cpus-per-task=5
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

module purge
module load cuda/11.3.1
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
source /home/sr5796/.bashrc
source activate torch_env
cd /scratch/sr5796/ML/gpt2_5000
python gpt2_trainer_5000.py
