#!/bin/sh
#SBATCH -J alphazero_training
#SBATCH -t 02:30:00
#SBATCH -n 1 -c 12
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=an0111ol-s@student.lu.se
#
python3 game.py
#script ends
