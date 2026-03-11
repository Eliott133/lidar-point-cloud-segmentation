# Lidar Point Cloud Segmentation

Ce dépôt contient notre projet de segmentation sémantique de nuages de points LiDAR.

Pour garantir une portabilité maximale et une installation ultra-rapide des dépendances, ce projet utilise **`uv`** comme gestionnaire d'environnement et de paquets Python.

---

## 📋 Prérequis

* Une machine avec un GPU NVIDIA.
* **CUDA Toolkit** (Testé avec la version 12.4).
* **`uv`** installé sur votre système. (Si ce n'est pas le cas : `curl -LsSf https://astral.sh/uv/install.sh | sh`).

---

## Installation de l'environnement (Guide pour le Cluster)

La compilation des opérations C++/CUDA personnalisées (bibliothèque `pointops`) requiert une configuration stricte, particulièrement sur les serveurs de calcul partagés pour éviter les dépassements de mémoire ou les erreurs de compilation.

### 1. Clonage et initialisation

Commencez par cloner le dépôt et synchroniser l'environnement. `uv` va lire le fichier `uv.lock` et recréer le dossier `.venv` avec les versions exactes de PyTorch et des autres dépendances en quelques secondes.

```bash
git clone [https://github.com/Eliott133/lidar-point-cloud-segmentation.git](https://github.com/Eliott133/lidar-point-cloud-segmentation.git)
cd lidar-point-cloud-segmentation/Pointcept_github

# Synchronisation ultra-rapide des paquets et création du .venv
uv sync
2. Configuration du compilateur CUDA
Sur le cluster, le compilateur NVIDIA (nvcc) ne se trouve pas toujours dans les chemins par défaut. Il faut indiquer manuellement à PyTorch où le trouver, et lui spécifier les architectures GPU cibles pour éviter l'erreur Unknown CUDA arch.

Bash
# 1. Définir le chemin exact vers CUDA (Ex: /opt/cuda/12.4 sur notre cluster)
export CUDA_HOME=/opt/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 2. Forcer la compatibilité avec les architectures GPU courantes (Turing, Ampere, Ada, Hopper)
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
3. Compilation de Pointops
Avertissement de Mémoire : La compilation C++ en parallèle consomme énormément de RAM. Si vous lancez l'installation sans limiter les cœurs, le processus plantera avec l'erreur Exit status 137 (Out of Memory).

Bash
# Limiter la compilation à 2 processus simultanés
export MAX_JOBS=2

# Se placer dans le dossier de la librairie
cd libs/pointops

# Lancer la compilation (prend environ 3 à 5 minutes)
uv run python setup.py install
4. Vérification de l'installation
Une fois la compilation terminée, revenez à la racine du projet et vérifiez que le module Python a bien été généré :

Bash
cd ../..
uv run python -c "import pointops; print('Succès : Pointops est correctement installé et opérationnel')"

Bash
# Activation de l'environnement
source .venv/bin/activate

## 🚀 Lancement de l'entraînement (Cluster SLURM)

Ce projet est configuré pour être exécuté sur un cluster de calcul via le gestionnaire de tâches **SLURM**.

Une fois l'environnement `uv` prêt et `pointops` compilé, vous pouvez lancer l'entraînement du modèle (par exemple PointTransformerV3) en soumettant le script batch fourni :

```bash
sbatch lancer_ptv3.sh
