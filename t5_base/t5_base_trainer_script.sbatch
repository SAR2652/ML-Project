#!/bin/bash
#SBATCH --job-name=t5_base_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1

module purge
module load cuda/11.3.1
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
source /home/sr5796/.bashrc
source activate torch_env
cd /scratch/sr5796/ML_Project
python t5_trainer.py
