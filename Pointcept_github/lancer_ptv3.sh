#!/bin/bash
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -J ptv3_kitti
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpue06
#SBATCH --time 150:00:00
#SBATCH --output=slurm_ptv3_%j.out
#SBATCH --error=slurm_ptv3_%j.err

module purge

# 1. On se place à la RACINE du projet Pointcept (CORRIGÉ)
cd /info/raid-etu/m2/s2400752/ProjetAlternance/Pointcept

# 2. On active l'environnement Python (Ton chemin est parfait)
source /info/etu/m2/s2400752/miniconda3/bin/activate pointcept

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================================="
echo "Dossier courant : $(pwd)"
echo "Début de l'entraînement sur le noeud : $(hostname)"
echo "GPU alloué :"
nvidia-smi
echo "=========================================================="

# 3. Lancement via le script officiel de Pointcept (CORRIGÉ)
# -p : trouve automatiquement ton python conda
# -g : 1 GPU
# -d : dataset semantic_kitti
# -c : fichier de configuration de base pour PTv3
# -n : le nom du dossier où seront sauvegardés tes résultats
sh scripts/train.sh -p $(which python) -g 1 -d semantic_kitti -c semseg-pt-v2m2-0-base -n run_ptv2_kitti
