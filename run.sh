#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 2
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eros.fani@studenti.polito.it
##
# load cuda module
module load nvidia/cudasdk/10.1
python -m torch.distributed.launch --nproc_per_node=1 run.py \
--num_workers 2 \
--dataset cts \
--name default \
--lr 2.5e-3 \
--epochs 360 \
--val_interval 30 \
--hnm
