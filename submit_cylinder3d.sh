#!/bin/bash
#SBATCH --job-name=cyl3d_kitti_v2
#SBATCH --output=logs/train_cyl3d_kitti_%j.out
#SBATCH --error=logs/train_cyl3d_kitti_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpue06

cd ./Cylinder3D

# chargement module cuda
module load cuda

nvcc --version
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Démarrage de l'entraînement via UV..."

uv run --project Cylinder3D torchrun --nproc_per_node=1 train_cylinder_asym.py --config_path ../configs/config_cylinder3d.yaml

# uv run --project Cylinder3D python test.py