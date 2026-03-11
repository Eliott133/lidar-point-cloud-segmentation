# Lidar Point Cloud Segmentation

Ce dépôt contient notre projet de segmentation sémantique de nuages de points LiDAR.

Pour garantir une portabilité maximale et une installation ultra-rapide des dépendances, ce projet utilise **`uv`** comme gestionnaire d'environnement et de paquets Python.

---

## 📋 Prérequis

* Une machine avec un GPU NVIDIA.
* **CUDA Toolkit** (Testé avec la version 12.4).
* **`uv`** installé sur votre système. (Si ce n'est pas le cas : `curl -LsSf https://astral.sh/uv/install.sh | sh`).

---

## Installation (À faire une seule fois)

L'environnement Python se configure tout seul. La seule étape requise lors du premier clonage est la compilation de la bibliothèque C++/CUDA personnalisée (pointops).

### 1. Clonage du projet


```bash
git clone https://github.com/Eliott133/lidar-point-cloud-segmentation.git
cd lidar-point-cloud-segmentation
```

### 2. Compilation de Pointops

Placez-vous dans le dossier du modèle, chargez les variables CUDA du serveur, et lancez la compilation. L'outil uv se chargera de créer l'environnement virtuel.

```bash
cd Pointcept

export CUDA_HOME=/opt/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Lancer la compilation (prend environ 3 à 5 minutes)
cd libs/pointops
uv run python setup.py install

# Revenir à la racine du projet
cd ../../../

```

### Lancement de l'entraînement

```bash
sbatch submit_pointcept.sh
```
