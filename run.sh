#!/bin/bash
#SBATCH --job-name=BSNV2_CTS
#SBATCH --time=99:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --partition=cuda
#SBATCH --gres=gpu:2
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eros.fani@gmail.com
##
# load cuda module
module load nvidia/cudasdk/10.1
python -m torch.distributed.launch --nproc_per_node=1 run.py \
--model bisenetv2 \
--num_workers 8 \
--dataset cts \
--name BSNV2_CTS \
--lr 5e-2 \
--weight_decay 0.0005 \
--epochs 690 \
--val_interval 30 \
--hnm \
--batch_size 16 \
--ignore_warnings \
--print \
--output_aux \
--no_pretrained
