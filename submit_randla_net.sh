#!/bin/bash
#SBATCH --job-name=randlanet_train
#SBATCH --output=logs/randlanet_kitti_%j.out
#SBATCH --error=logs/randlanet_kitti_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpue06

set -e

echo "===================================="
echo "Job started on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "===================================="

REPO_DIR=./RandLA-Net-pytorch

cd "$REPO_DIR"

export CUDA_HOME=/opt/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH

nvidia-smi

echo "===================================="
echo "STEP 1: CHECK PREPROCESSED DATA"
echo "===================================="

ls /info/raid-etu/m2/s2405735/pretrainned | head
ls /info/raid-etu/m2/s2405735/pretrainned/sequences_0.06 | head

uv run sh compile_op.sh

echo "===================================="
echo "STEP 2: START TRAINING"
echo "===================================="

uv run python main_SemanticKITTI.py --log_dir logs --max_epoch 150 --batch_size 2
echo "===================================="
echo "TRAINING FINISHED"
echo "===================================="

date
