from torch.utils.data import DataLoader
from torchvision import transforms


from datasets.semantic_kitti import SemanticKittiDataset
from datasets.collate import custom_collate_fn
from datasets.transforms import SemanticKittiLabelRemap, ToTensor, RandomRotationZ

def main():
    CHEMIN_DATASET = "/info/corpus/SemanticKITTI/dataset" 
    
    # Pour l'entraînement : Remap -> Rotation Aléatoire -> Tenseur
    train_transforms = transforms.Compose([
        SemanticKittiLabelRemap(),
        RandomRotationZ(probability=0.8), # On l'applique 80% du temps
        ToTensor()
    ])
    
    # Pour la validation : Remap -> Tenseur
    val_transforms = transforms.Compose([
        SemanticKittiLabelRemap(),
        ToTensor()
    ])
    
    train_dataset = SemanticKittiDataset(
        root_dir=CHEMIN_DATASET,
        sequences=['00', '01'], 
        transform=train_transforms, # Pipeline avec augmentation
        load_labels=True
    )
    
    val_dataset = SemanticKittiDataset(
        root_dir=CHEMIN_DATASET,
        sequences=['08'],           # Séquence officielle de validation
        transform=val_transforms,   # Pipeline standard
        load_labels=True
    )
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True, # On mélange pour l'entraînement
        collate_fn=custom_collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False, # Pas besoin de mélanger la validation
        collate_fn=custom_collate_fn
    )

    print(f"Dataset prêt : {len(train_dataset)} scans d'entraînement, {len(val_dataset)} scans de validation.")

if __name__ == "__main__":
    main()