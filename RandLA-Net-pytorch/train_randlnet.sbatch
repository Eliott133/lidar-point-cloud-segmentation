#!/bin/bash
#SBATCH --job-name=randlanet_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
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

REPO_DIR=/info/raid-etu/m2/s2405735/projet_2sem/RandLA-Net-pytorch
PYTHON_BIN=/info/etu/m2/s2405735/miniconda3/envs/randlanet/bin/python

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}" || exit 1

echo "Python: ${PYTHON_BIN}"
echo "Repo: ${REPO_DIR}"

echo "========= GPU INFO ========="
nvidia-smi

"${PYTHON_BIN}" -c "import sys; print('Python:', sys.executable)"
"${PYTHON_BIN}" -c "import torch; print('Torch:', torch.__version__)"


echo "===================================="
echo "STEP 2: CHECK PREPROCESSED DATA"
echo "===================================="

ls /info/raid-etu/m2/s2405735/pretrainned | head
ls /info/raid-etu/m2/s2405735/pretrainned/sequences_0.06 | head

echo "===================================="
echo "STEP 3: START TRAINING"
echo "===================================="

"${PYTHON_BIN}" main_SemanticKITTI.py

echo "===================================="
echo "TRAINING FINISHED"
echo "===================================="

date