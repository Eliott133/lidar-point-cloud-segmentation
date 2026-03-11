
import numpy as np
import torch

class SemanticKittiLabelRemap(object):
    """
    Transforme les labels bruts de SemanticKITTI en 19 classes d'entraînement valides.
    Utilise une table de correspondance (lookup table) NumPy pour des performances optimales.
    """
    def __init__(self):
        self.learning_map = {
            0 : 0,     1 : 0,     10: 1,     11: 2,     13: 5,     15: 3,     
            16: 5,     18: 4,     20: 5,     30: 6,     31: 7,     32: 8,     
            40: 9,     44: 10,    48: 11,    49: 12,    50: 13,    51: 14,    
            52: 0,     60: 9,     70: 15,    71: 16,    72: 17,    80: 18,    
            81: 19,    99: 0,     252: 1,    253: 2,    254: 6,    255: 8,    
            256: 5,    257: 5,    258: 4,    259: 5
        }
        max_key = max(self.learning_map.keys())
        self.lookup_table = np.zeros((max_key + 1), dtype=np.int32)
        for k, v in self.learning_map.items():
            self.lookup_table[k] = v

    def __call__(self, sample):
        if 'semantics' in sample:
            raw_labels = sample['semantics']
            safe_labels = np.clip(raw_labels, 0, len(self.lookup_table) - 1)
            sample['semantics'] = self.lookup_table[safe_labels]
        return sample

class ToTensor(object):
    """
    Convertit les arrays NumPy en Tenseurs PyTorch.
    """
    def __call__(self, sample):
        sample['points'] = torch.tensor(sample['points'], dtype=torch.float32)
        sample['remissions'] = torch.tensor(sample['remissions'], dtype=torch.float32)
        if 'semantics' in sample:
            # Les labels de classification doivent être en format Long (int64)
            sample['semantics'] = torch.tensor(sample['semantics'], dtype=torch.long)
            sample['instances'] = torch.tensor(sample['instances'], dtype=torch.long)
        return sample
    
class RandomRotationZ(object):
    """
    Applique une rotation aléatoire autour de l'axe Z (vertical) au nuage de points.
    Probabilité par défaut de 50%.
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        # On tire un nombre au hasard pour savoir si on applique la rotation
        if np.random.rand() < self.probability:
            points = sample['points']
            
            # Générer un angle aléatoire entre 0 et 2*Pi (0 et 360 degrés)
            angle = np.random.uniform(0, 2 * np.pi)
            
            cosval = np.cos(angle)
            sinval = np.sin(angle)
            
            # Création de la matrice de rotation Z
            rotation_matrix = np.array([
                [cosval, -sinval, 0],
                [sinval,  cosval, 0],
                [0,       0,      1]
            ], dtype=np.float32)
            
            # Produit matriciel : (N, 3) @ (3, 3) -> (N, 3)
            # On utilise le symbole '@' de Python pour la multiplication de matrices
            rotated_points = points @ rotation_matrix
            
            # On met à jour les points dans le dictionnaire
            sample['points'] = rotated_points
            
        return sample