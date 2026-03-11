# lidar-point-cloud-segmentation

![Cover](https://camo.githubusercontent.com/81683611ce8e8f07c3627476bf3e18f8193ff24603208d9fbdbc546fa8a72d4d/68747470733a2f2f696d6167652e6962622e636f2f6b79684372562f7363616e312e706e67)

> [Source de l'image](https://github.com/PRBonn/semantic-kitti-api)

## Présentation

Ce projet est dédié à la segmentation sémantique 3D de nuages de points LiDAR à grande échelle, en utilisant principalement le dataset SemanticKITTI. L'objectif est d'assigner une classe sémantique (véhicule, piéton, route, etc.) à chaque point individuel d'un scan LiDAR.

Le dépôt regroupe plusieurs architectures de pointe pour comparer leurs performances et leur efficacité : PointNet, RandLA-Net, Cylinder3D et la suite de modèles via Pointcept.

## Contributeurs

| [Eliott](https://github.com/Eliott133) | [Josué](https://github.com/JosueAhobade) | [Raphael](https://github.com/Rapha76) |
| --- | --- | --- |


## Dataset

### Overview
Ce projet utilise **SemanticKITTI**, un dataset de référence à grande échelle pour la compréhension de scènes en extérieur via LiDAR. Basé sur le célèbre dataset KITTI Vision Benchmark, SemanticKITTI fournit des annotations denses point par point pour l'intégralité des séquences d'odométrie LiDAR. Il couvre une grande variété d'environnements urbains (rues, autoroutes, zones résidentielles) et inclut des objets statiques ainsi que des agents dynamiques.

### Objectif
L'objectif associé à ce dataset est la **segmentation sémantique 3D**. Pour chaque scan LiDAR (nuage de points), le modèle d'apprentissage profond (ex: RandLA-Net, PointPillars, etc.) doit prédire une classe sémantique spécifique (voiture, piéton, route, bâtiment, végétation, etc.) pour **chaque point individuel** de la scène.

### Structure du Dossier
Pour que les scripts d'entraînement et de prétraitement fonctionnent correctement, le dataset doit être organisé selon la hiérarchie suivante :

```text
dataset/
 └── sequences/               # Contient l'ensemble des séquences de conduite (00 à 21)
     ├── 00/                  # Séquence de conduite n°00 (identique pour 01, 02, etc.)
     │   ├── poses.txt        # Poses du véhicule 3D (matrices de transformation) pour chaque frame
     │   ├── times.txt        # Horodatages (timestamps) exacts de chaque scan LiDAR
     │   ├── calib.txt        # Matrices de calibration des capteurs (caméras, LiDAR)
     │   │
     │   ├── velodyne/        # Nuages de points LiDAR bruts
     │   │   ├ 000000.bin     # Scan n°0 (coordonnées X,Y,Z et réflectance en float32)
     │   │   └ 000001.bin     # Scan n°1...
     │   │
     │   ├── labels/          # Annotations sémantiques pour la segmentation de nuages de points
     │   │   ├ 000000.label   # Labels associés au scan velodyne/000000.bin (classes et instances)
     │   │   └ 000001.label   # Labels du scan n°1...
     │   │
     │   └── voxels/          # Données pour la tâche de Semantic Scene Completion (grilles 3D denses)
     │       ├ 000000.bin     # Grille de voxels binaire (géométrie de l'espace)
     │       ├ 000000.label   # Labels sémantiques pour chaque voxel de la grille
     │       ├ 000000.occluded # Masque indiquant les voxels cachés par d'autres objets
     │       ├ 000000.invalid # Masque des voxels hors du champ de vision ou non évaluables
     │       ├ 000001.bin
     │       ├ 000001.label
     │       ├ 000001.occluded
     │       └ 000001.invalid
     │
     ├── 01/                  # Séquence suivante...
     ├── 02/
     .
     .
     └── 21/
```

Pour la segmentation sementique 3D nous utilisons les dossiers : `velodyne/` et `labels/`

### Taille et Volumétrie
SemanticKITTI est un dataset à très grande échelle. Avant de lancer le téléchargement ou les scripts d'entraînement, assurez-vous de disposer de suffisamment d'espace de stockage et de mémoire matérielle (RAM / VRAM).

* **Nombre de scans (Frames)** : **43 552 scans LiDAR** au total (23 201 pour l'entraînement/validation, et 20 351 pour le test).
* **Nombre de points** : Un scan Velodyne typique contient entre 100 000 et 130 000 points. Le dataset complet représente **plus de 4,5 milliards de points annotés** individuellement.
* **Espace disque requis (Estimations)** :
  * Nuages de points bruts (`velodyne/`) : ~75 Go
  * Annotations (`labels/`) : ~1 Go
  * Grilles denses (`voxels/` - *uniquement si utilisé*) : ~10 Go supplémentaires.
  * **Recommandation** : Prévoyez au moins **90 Go à 110 Go d'espace libre** sur un disque de stockage très rapide (idéalement un SSD NVMe) pour décompresser l'intégralité du dataset et héberger les fichiers temporaires/cache générés lors des prétraitements.

### Format des données 

- **Nuages de points (`.bin`)** : Les données LiDAR brutes sont stockées sous forme de fichiers binaires. Chaque point est représenté par 4 valeurs en virgule flottante `(float32)` : les coordonnées spatiales `(X, Y, Z)` et la réflectance/intensité du signal LiDAR.

- **Labels (`.label`)** : Les annotations sont stockées dans des fichiers binaires correspondants. Chaque point possède un label de 32 bits (``uint32``), où les 16 bits de poids faible représentent l'ID de la classe sémantique, et les 16 bits de poids fort représentent l'ID de l'instance (utile pour la segmentation d'instance).

### Découpage du Dataset (Data Splits)

Le dataset est composé de 22 séquences LiDAR consécutives, réparties de manière standard pour permettre une évaluation équitable :

- Set d'Entraînement (Train) : Séquences ``00`` à ``07``

- Set de Validation (Val) : Séquence ``08`` (environ 4 000 scans).

- Set de Test (Test) : Séquences ``11`` à ``21`` (environ 20 000 scans).

> Note : Les labels de test ne sont pas publics et l'évaluation doit se faire via le serveur officiel.

### Classes

L'ensemble de données contient 28 classes, dont certaines distinguent les objets immobiles des objets mobiles. Dans l'ensemble, les classes couvrent les usagers de la route, mais aussi des classes fonctionnelles pour le sol, comme les aires de stationnement et les trottoirs.

![Distribution classes](https://semantic-kitti.org/images/label_distribution.svg)

> [Source de l'image](https://semantic-kitti.org/dataset.html)

`configs/semantic-kitti.yaml` définit la manière dont les données du dataset SemanticKITTI sont interprétées, traitées et évaluées. On y retrouve :
- Définition des labels
- Carte des couleurs
- Statistiques de contenu
- Mapping d'apprentissage
- Mapping inverse et ignorés

### Development Kit (DevKit)
Un kit de développement officiel (API) est proposé pour faciliter la manipulation de ce dataset. Ce DevKit inclut divers outils et scripts permettant de :

- Visualiser les nuages de points 3D et leurs annotations

- Évaluer les performances en calculant des métriques standards comme le mIoU de manière officielle.

- Mapping des labels.

[Repo du DevKit SementicKITII](https://github.com/PRBonn/semantic-kitti-api)

###  Lien de Téléchargement
Les données doivent être téléchargées depuis le site officiel :

[Lien dataset SemanticKITTI](https://semantic-kitti.org/dataset.html#download)

Assurez-vous de télécharger le fichier data_odometry_velodyne.zip et data_odometry_labels.zip, puis de les extraire et de les fusionner dans le dossier sequences/.

### Licence
Le dataset KITTI original (nuages de points) est sous licence [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)](https://creativecommons.org/licenses/by-nc-sa/3.0/).

Les annotations SemanticKITTI sont sous licence [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**Ces données sont fournies uniquement pour un usage académique et de recherche non commerciale.**

## Installation

Le projet utilise ``uv`` pour la gestion des dépendances et de l'environnement Python.

**Prérequis :**

- CUDA Toolkit (compatible avec PyTorch)
- [uv | astral (uv 0.9.30)](https://docs.astral.sh/uv/getting-started/installation/)
- Vous disposez des GPU de la famille RTX (RTX 2080, RTX 6000, RTX 8000)
`

1) Cloner le projet sur le cluster de l'université du mans

```bash
git clone https://github.com/Eliott133/lidar-point-cloud-segmentation.git
```

2) Vérifier que uv est installer 

```bash
uv --version
```

3) Vous pouvez lancer l'entrainement des modèles en lancant les scripts :

```bash
sbatch submit_cylinder3d.sh # Pour lancer l'entrainement du modèle Cylinder3D
sbatch submit_pointcept.sh # Pour lancer l'entrainement du modèle PointCept
sbatch submit_pointnet.sh # # Pour lancer l'entrainement du modèle PointNet
sbatch submit_randla_net.sh # # Pour lancer l'entrainement du modèle RandLa-Net
```

Ces scripts lancent un job sur la machine gpue06 et crée l'environnement adéquat directement

