#!/bin/bash
#SBATCH --job-name=cyl3d_kitti_v2
#SBATCH --output=logs/train_cyl3d_kitti_%j.out
#SBATCH --error=logs/train_cyl3d_kitti_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --partition=gpu

cd ./Cylinder3D

# 2. Charger les modules système nécessaires (CUDA)
module load cuda

# 3. Vérifications de routine (Optionnel)
nvcc --version
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Lancer l'entraînement avec uv
echo "Démarrage de l'entraînement via UV..."

# uv run --project Cylinder3D torchrun --nproc_per_node=1 train_cylinder_asym.py --config_path config/semantickitti.yaml

uv run --project Cylinder3D python test.py