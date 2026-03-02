#!/bin/bash
#SBATCH --job-name=lidar_seg_m2        # Nom de ton job
#SBATCH --output=logs/resultat_%j.out  # Fichier où s'écrira ton "print()" ( %j = numéro du job)
#SBATCH --error=logs/erreur_%j.err     # Fichier où s'écriront les erreurs
#SBATCH --time=40:00:00                # Temps max alloué (Heures:Minutes:Secondes)
#SBATCH --partition=gpue06             # Nom de la partition GPU (à vérifier avec ta fac)
#SBATCH --gres=gpu:rtx8000:1           # Demande de 1 carte RTX 8000 (ou rtx6000)
#SBATCH --cpus-per-task=20             # Nombre de cœurs CPU (pour charger les données vite)
#SBATCH --mem=100G                      # Mémoire RAM classique (hors GPU)

echo "Début du job sur le noeud : $HOSTNAME"

STOCKAGE_DIR="/info/raid-etu/m2/i2101721"
NOEUD_LOCAL_DIR=$TMPDIR/dataset_lidar 

echo "Copie des données du stockage vers le disque local NVMe..."
mkdir -p $NOEUD_LOCAL_DIR
tar -xzf $STOCKAGE_DIR/semantic_kitti_mini.tar.gz -C $NOEUD_LOCAL_DIR
echo "Copie terminée !"


echo "Lancement du script Python..."
python train_ou_inference.py --data_path $NOEUD_LOCAL_DIR --output_dir $NOEUD_LOCAL_DIR/resultats

echo "Sauvegarde des résultats vers le cluster de stockage..."
cp -r $NOEUD_LOCAL_DIR/resultats $STOCKAGE_DIR/mes_resultats_finaux/

echo "Job terminé avec succès !"
