#!/bin/bash
#SBATCH --job-name=pointnet_kitti_v2
#SBATCH --output=logs/pointnet_kitti_%j.out
#SBATCH --error=logs/pointnet_kitti_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpue06

cd ./PointNet

# chargement module cuda
module load cuda/12.4

nvcc --version
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Démarrage de l'entraînement via UV..."

uv run --project PointNet torchrun --nproc_per_node=1 train_pointnet.py --config_path ../configs/config_pointnet.yaml