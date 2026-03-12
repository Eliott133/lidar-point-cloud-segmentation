#!/bin/bash
#SBATCH --job-name=ptv3_kitti
#SBATCH --output=logs/train_ptv3_kitti_%j.out
#SBATCH --error=logs/train_ptv3_kitti_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpue06

cd $SLURM_SUBMIT_DIR

cd ./Pointcept

module purge
module load cuda

export CUDA_HOME=/opt/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH

nvidia-smi

echo " Vérification de la bibliothèque C++ (Pointops."

if ! uv run python -c "import pointops" &> /dev/null; then
    echo " Pointops non trouvé. Compilation en cours (3-5 minutes."
    
    # Variables spécifiques pour éviter que le cluster ne plante par manque de RAM
    export MAX_JOBS=2
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    
    cd libs/pointops
    uv run python setup.py install
    cd ../..
    
    echo " Compilation de Pointops terminée avec succs"
else
    echo " Pointop trouvé."
fi

echo "Démarrage de l'entraînement."

uv run python tools/train.py --config-file ../configs/config_pointcept.py --options save_path=exp/semantic_kitti/run_ptv2_kitti
