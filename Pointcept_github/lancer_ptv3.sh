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

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

export CUDA_HOME=/opt/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================================="
echo "Dossier courant : $(pwd)"
echo "Début de l'entraînement sur le noeud : $(hostname)"
echo "GPU alloué :"
nvidia-smi
echo "=========================================================="

# -p : trouve automatiquement ton python conda
# -g : 1 GPU
# -d : dataset semantic_kitti
# -c : fichier de configuration de base pour PTv3
# -n : le nom du dossier où seront sauvegardés tes résultats
sh scripts/train.sh -p $(which python) -g 1 -d semantic_kitti -c semseg-pt-v2m2-0-base -n run_ptv2_kitti
