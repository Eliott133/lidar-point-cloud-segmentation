# Architectures de Deep Learning : Détection 2D et Nuages de Points 3D

Ce document référence et détaille plusieurs architectures incontournables pour la vision par ordinateur, allant de la détection d'objets en 2D à la segmentation et compréhension spatiale des nuages de points 3D.

---

## 1. PointNet

### Types de Couches

* **MLP (Convolutions partagées)** : Chaque point passe par les mêmes poids de réseau, indépendamment des autres points.
* **Réseaux d'Alignement (T-Net)** : Des mini-réseaux intégrés qui prédisent des matrices de transformation (affinages) appliquées directement aux coordonnées ou aux caractéristiques pour aligner l'objet dans l'espace.
* **Max Pooling Global** : Une opération de tri qui extrait la valeur maximale de chaque dimension sur l'ensemble des points. Cela crée une signature unique (vecteur global) totalement indépendante de l'ordre d'entrée des points.

### Flux de Données

Pour un nuage de points en entrée de N points avec 3 coordonnées (X, Y, Z) :

1. **Entrée** : Matrice de dimension [N, 3].
2. **Input Transform (T-Net)** : Aligne spatialement les points bruts (Matrice 3×3).
3. **Extraction locale 1** : MLP partagés (tailles 64, 64) ➔ Sortie de dimension [N, 64].
4. **Feature Transform (T-Net)** : Aligne les caractéristiques extraites (Matrice 64×64).
5. **Extraction locale 2** : MLP partagés (tailles 64, 128, 1024) ➔ Sortie de dimension [N, 1024].
6. **Agrégation Globale** : Couche de Max Pooling sur la dimension N ➔ Signature globale de dimension [1, 1024].
7. **Classification** : Réseau dense classique (MLP de tailles 512, 256, Dropout) puis couche finale correspondant au nombre K de classes ➔ Sortie finale [1, K].

### Taille du Modèle et Performances

| Caractéristique | Détail |
| --- | --- |
| **Paramètres** | ~3,5 millions |
| **Poids du fichier** | ~14 Mo (très léger) |
| **Vitesse** | Très rapide (> 1 million de points/sec sur GPU classique), adapté au temps réel pour véhicules autonomes. |

### Forces et Limites

* **Forces** : Résiste au désordre des points, gère un nombre variable de points en entrée, très rapide, peu gourmand en mémoire.
* **Limites** : Traite chaque point de manière isolée avant l'agrégation finale. Il ne capture donc pas bien les structures géométriques locales (les relations entre les points voisins).

---

## 2. PointNet++

PointNet++ a été proposé pour pallier les insuffisances de PointNet (traitement indépendant des points, utilisation d'un seul Max Pooling global, absence de capture de la structure locale).

### Architecture Codeur-Décodeur

**Encodeur : Set Abstraction (SA)**
Chaque couche SA comprend 3 étapes pour sous-échantillonner et extraire des caractéristiques :

1. **Sampling (Échantillonnage)** : Utilise la méthode *Farthest Point Sampling* (FPS) pour sélectionner un sous-ensemble représentatif de points et réduire progressivement la taille du nuage.
2. **Grouping (Voisinage)** : Pour chaque point échantillonné, on recherche ses voisins (via kNN ou *ball query* par rayon) pour créer des patchs locaux 3D.
3. **Mini-PointNet local** : Sur chaque voisinage, application d'un MLP partagé et d'un Max Pooling local.

**Décodeur : Feature Propagation (FP)**
Pour les tâches de prédiction dense (comme la segmentation sémantique), le modèle doit récupérer une prédiction par point à la résolution d'origine.

1. **Interpolation** : Les caractéristiques de niveau supérieur (moins de points, contexte plus global) sont transférées vers des niveaux plus denses via une interpolation pondérée inversement par la distance aux points proches.
2. **Skip Connections (Connexions résiduelles)** : Les caractéristiques interpolées sont fusionnées avec celles conservées par l'encodeur au même niveau. Cela récupère les détails locaux fins tout en conservant le contexte global.
3. **MLP partagé** : Les données fusionnées sont affinées via un petit MLP pour mettre à jour les représentations. Le processus est répété jusqu'à la résolution initiale.

### Flux de Données (Segmentation)

1. **Entrée** : Nuage de points [N, 3].
2. **Descente (SA)** : N points ➔ SA1 (ex: 1024 points) ➔ SA2 (ex: 256 points) ➔ SA3 (ex: 64 points avec contexte global).
3. **Remontée (FP)** : FP3 (interpolation vers 256 pts + fusion) ➔ FP2 (vers 1024 pts) ➔ FP1 (vers N pts).
4. **Sortie** : Prédiction de classe pour chaque point d'origine [N, K].

### Taille du Modèle et Performances

| Caractéristique | Détail |
| --- | --- |
| **Paramètres** | ~1,48 millions (pour la classification) / ~0,97 millions (segmentation) |
| **Poids du fichier** | ~6 à 10 Mo |
| **Vitesse** | Plus lent que PointNet à cause de l'échantillonnage FPS et de la recherche de voisins. |

### Forces et Limites

* ** Forces** : Apprend des représentations robustes et détaillées à plusieurs échelles, excellentes performances pour la compréhension de scènes 3D complexes.
* ** Limites** : Coût computationnel élevé, scalabilité limitée sur d'immenses nuages de points (le calcul du FPS devient un goulot d'étranglement), grande sensibilité aux hyperparamètres.

---

## 3. Faster R-CNN

Cette architecture est la référence pour la détection d'objets en 2D (génération de *bounding boxes*).

###  Architecture

* **Region Proposal Network (RPN)** : Un réseau CNN qui propose des régions susceptibles de contenir un objet. Ces propositions se basent sur des **ancres** (*anchors*), qui sont des boîtes prédéfinies (par défaut 3 échelles et 3 ratios).
* **Fast R-CNN** : Le module de détection qui partage les mêmes couches de convolution que le RPN. Il récupère les régions proposées pour classifier l'objet et ajuster précisément la taille de la boîte (régression).

###  Flux de Données

1. **Extraction (Backbone)** : L'image d'entrée traverse un réseau convolutif pour générer une *feature map* de dimensions W × H.
2. **Fenêtre glissante (RPN)** : Une convolution 3×3 parcourt la *feature map* pour mapper l'information vers un tenseur intermédiaire.
3. **Ancres et Prédictions (RPN)** : Pour chaque position spatiale, 9 ancres sont générées. Le tenseur passe ensuite par deux couches sœurs :
* *Couche de classification (cls)* : Produit 18 scores (2 classes × 9 ancres) pour estimer la probabilité Objet vs Fond.
* *Couche de régression (reg)* : Produit 36 coordonnées (4 décalages × 9 ancres) pour ajuster le centre, la largeur et la hauteur.


4. **Filtrage NMS** : Algorithme de *Non-Maximum Suppression* pour éliminer les boîtes redondantes. Si une boîte en chevauche une autre avec un score inférieur à plus de 70% (IoU > 0.7), la moins pertinente est supprimée.
5. **Détection Finale** : Fast R-CNN projette les boîtes filtrées sur la *feature map* partagée pour la classification finale et l'ajustement au pixel près.

### Taille du Modèle et Entraînement

| Caractéristique | Détail |
| --- | --- |
| **Paramètres** | ~136 millions (avec VGG-16) ou ~60 millions (avec ZF Net) |
| **Entraînement RPN** | Ancre positive si IoU > 0.7 avec la vérité terrain, négative si < 0.3. Loss combinée (classification + régression). |

### Forces et Limites

* ** Forces** : Très rapide en inférence comparé à ses prédécesseurs (R-CNN et Fast R-CNN) grâce au partage strict des calculs convolutifs.
* ** Limites** : L'entraînement est complexe (optimisations alternées). Modèle assez lourd selon le *backbone* choisi, ce qui peut freiner un déploiement sur de l'embarqué très léger.

---

## 4. PointPillars

### Types de Couches

* **Pillar Feature Network (PFN)** : Mini-PointNet (MLP + Conv1D) appliqué localement à l'intérieur de chaque "pilier" (colonne verticale découpant l'espace 3D) pour extraire les caractéristiques des points.
* **Max Pooling** : Agrège les informations des points d'un même pilier pour en faire un tenseur dense.
* **Convolutions 2D (Backbone)** : Une fois les piliers dispersés (*scatter*) sur une grille 2D (créant une "pseudo-image" vue de dessus), le réseau utilise des convolutions 2D standards, beaucoup plus rapides que des convolutions 3D.
* **Tête SSD (Single Shot Detector)** : Couches finales 2D chargées de prédire les boîtes englobantes 3D (centre X, Y, Z, dimensions, angle de rotation).

### Flux de Données

1. **Voxelisation en piliers** : Le nuage de points continu est découpé en une grille x-y. Chaque case de la grille forme une colonne infinie (un pilier) sur l'axe z.
2. **Extraction PFN** : Les points dans chaque pilier passent par un MLP partagé suivi d'un Max Pooling ➔ on obtient un vecteur de caractéristiques par pilier.
3. **Génération de la Pseudo-Image** : Les vecteurs de caractéristiques sont replacés dans leur position x-y d'origine, formant une image 2D multi-canaux (Bird's Eye View).
4. **Traitement 2D** : Cette pseudo-image passe dans un réseau de convolution 2D classique (similaire à un détecteur d'image).
5. **Prédiction SSD** : Le réseau de sortie prédit les boîtes de détection 3D directement depuis les *feature maps* 2D.

### Taille du Modèle et Paramètres

| Caractéristique | Détail |
| --- | --- |
| **Paramètres** | ~4,8 millions |
| **Poids du fichier** | ~15 à 20 Mo |
| **Vitesse** | Ultra-rapide (souvent > 60 FPS), conçu pour le traitement lidar temps-réel en conduite autonome. |

### Forces et Limites

* ** Forces** : Vitesse d'exécution exceptionnelle en évitant les coûteuses convolutions 3D. Très bonne détection des véicules et objets volumineux.
* ** Limites** : En écrasant l'axe Z (la hauteur), le modèle perd certaines informations de géométrie fine. Moins précis sur les petits objets complexes (piétons, cyclistes ou feux de signalisation suspendus).

---

## 5. RandLA-Net

### Types de Couches et Architecture

Les méthodes classiques (comme le *Farthest Point Sampling* de PointNet++) sont un goulot d'étranglement en termes de temps de calcul sur d'immenses scènes (ex: lidar extérieur avec 1 million de points). RandLA-Net repose sur **l'échantillonnage aléatoire** (*Random Sampling*), qui est computationnellement gratuit $O(1)$.

Pour compenser la perte potentielle de points importants lors de ce tirage aléatoire, RandLA-Net utilise un module de **Local Feature Aggregation (LFA)**, composé de :

* **Local Spatial Encoding (LocSE)** : Encode explicitement les coordonnées x, y, z de chaque point et de ses voisins kNN pour préserver la géométrie spatiale complexe.
* **Attentive Pooling** : Remplace le Max Pooling classique. Un réseau neuronal attribue des "poids d'attention" à chaque voisin, permettant de conserver uniquement les caractéristiques les plus pertinentes.
* **Dilated Residual Block** : Empile plusieurs modules LFA pour élargir massivement le champ réceptif sans alourdir le calcul.

### Flux de Données

1. **Entrée** : Nuage de points massif $[N, d]$ (où $d$ est la dimension : coordonnées + couleur/réflectance).
2. **Encodeur (Descente)** :
* Application du module LFA sur les points.
* Échantillonnage aléatoire divisant le nombre de points par 4 à chaque étape ($N \rightarrow N/4 \rightarrow N/16$, etc.).


3. **Décodeur (Remontée)** : Utilise l'interpolation kNN pour sur-échantillonner les caractéristiques et des *skip-connections* pour retrouver la résolution initiale $N$.
4. **Sortie** : Couches MLP finales (Shared MLP) prédisant la classe sémantique de chaque point.

### Taille du Modèle et Performances

| Caractéristique | Détail |
| --- | --- |
| **Paramètres** | ~1,2 millions |
| **Poids du fichier** | ~5 Mo (extrêmement compact) |
| **Vitesse** | Peut traiter 1 million de points en une seule passe, jusqu'à 200x plus rapide que les méthodes basées sur des graphes (GCN) ou PointNet++. |

### Forces et Limites

* ** Forces** : Scalabilité massive (taille du nuage de points). Haute efficacité en mémoire et en temps de calcul. Surpasse les méthodes de l'état de l'art sur des benchmarks gigantesques (Semantic3D, SemanticKITTI).
* ** Limites** : L'échantillonnage aléatoire, bien qu'atténué par le module d'attention, peut théoriquement supprimer de minuscules objets isolés dans la scène avant qu'ils ne soient analysés en profondeur.
