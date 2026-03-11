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

echo "========= PYTHON INFO ========="
uv run --active python -c "import sys; print('Python:', sys.executable)"
uv run --active python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "========= GPU INFO ========="
nvidia-smi

echo "===================================="
echo "STEP 1: CHECK PREPROCESSED DATA"
echo "===================================="

ls /info/raid-etu/m2/s2405735/pretrainned | head
ls /info/raid-etu/m2/s2405735/pretrainned/sequences_0.06 | head

echo "===================================="
echo "STEP 2: START TRAINING"
echo "===================================="

uv run --active python main_SemanticKITTI.py \
  --checkpoint_path output/checkpoint.tar \
  --log_dir logs \
  --max_epoch 150 \
  --batch_size 2
echo "===================================="
echo "TRAINING FINISHED"
echo "===================================="

date
