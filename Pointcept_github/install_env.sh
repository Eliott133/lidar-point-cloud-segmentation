#!/bin/bash
uv sync
source .venv/bin/activate
pip uninstall -y torch-scatter torch-cluster torch-sparse
pip install --no-cache-dir torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

export CUDA_HOME=/opt/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
export MAX_JOBS=2
cd libs/pointops
python setup.py install
cd ../..

mkdir -p data
# On fait pointer SemanticKITTI directement à la bonne place
ln -s /info/corpus/SemanticKITTI data/semantic_kitti
