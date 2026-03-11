import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# --- Import de vos modules locaux ---
from datasets.semantic_kitti import SemanticKittiDataset
from datasets.transforms import SemanticKittiLabelRemap, ToTensor, RandomRotationZ, FixedPointSampler
from pointnet import PointNetSegmentation 

def feature_transform_regularizer(trans):
    """
    Régularisation pour le T-Net : force la matrice à rester proche d'une matrice orthogonale.
    Empêche le réseau d'aplatir le nuage de points.
    """
    d = trans.size()[1]
    I = torch.eye(d, requires_grad=True).to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def main(args):
    # INITIALISATION ET CONFIGURATION
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    train_cfg = config['training']
    data_cfg = config['dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lancement de l'entraînement sur : {device}")
    
    # Préparation du dossier de sauvegarde
    save_dir = Path(train_cfg.get('model_save_dir', './checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Les modèles seront sauvegardés dans : {save_dir} (Session: {run_timestamp})")


    # PIPELINES DE DONNÉES
    # Pipeline d'entraînement AVEC Data Augmentation
    train_transforms = transforms.Compose([
        SemanticKittiLabelRemap(),
        FixedPointSampler(num_points=train_cfg['num_points']), # Fixe la taille pour PointNet
        RandomRotationZ(probability=0.5),                      # Augmentation
        ToTensor()
    ])

    # Pipeline de validation SANS Data Augmentation
    val_transforms = transforms.Compose([
        SemanticKittiLabelRemap(),
        FixedPointSampler(num_points=train_cfg['num_points']), # Indispensable aussi ici pour faire des batchs !
        ToTensor()
    ])

    print("\nChargement des datasets...")
    train_dataset = SemanticKittiDataset(
        root_dir=data_cfg['root_dir'],
        sequences=data_cfg['train_sequences'], 
        transform=train_transforms,
        load_labels=True
    )
    
    val_dataset = SemanticKittiDataset(
        root_dir=data_cfg['root_dir'],
        sequences=data_cfg['val_sequences'], 
        transform=val_transforms,
        load_labels=True
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Un des datasets est vide Vérifiez le chemin : {data_cfg['root_dir']}")
    print(f"Datasets chargés : {len(train_dataset)} scans (Train) | {len(val_dataset)} scans (Val)")

    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=4)

    # MODÈLE, OPTIMISEUR ET LOSS
    model = PointNetSegmentation(config_path=args.config_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0) # On ignore la classe 'unlabeled'

    # Initialisation du traqueur pour sauvegarder le meilleur modèle
    best_val_loss = float('inf')

    # BOUCLE D'ENTRAÎNEMENT PRINCIPALE
    print("\nDébut de l'entraînement...")
    for epoch in range(train_cfg['epochs']):
        print(f"\n{'-'*30}\nEpoch {epoch+1}/{train_cfg['epochs']}\n{'-'*30}")
        
        # TRAIN
        model.train() # Active le Dropout et la mise à jour des BatchNorms
        train_loss = 0.0
        
        train_bar = tqdm(train_dataloader, desc="[Train]")
        for batch in train_bar:
            # Préparation des tenseurs (B, C, N) pour PointNet
            points = batch['points'].transpose(1, 2).to(device) 
            labels = batch['semantics'].to(device) 
            
            optimizer.zero_grad()
            preds, trans_feat = model(points) 
            
            # Calcul de la Loss (Segmentation + Régularisation)
            loss_seg = criterion(preds, labels)
            loss_reg = feature_transform_regularizer(trans_feat)
            loss = loss_seg + train_cfg['reg_weight'] * loss_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_dataloader)

        # VALIDATION
        model.eval() # Désactive le Dropout et fige les BatchNorms
        val_loss = 0.0
        
        val_bar = tqdm(val_dataloader, desc="[Val]  ")
        
        with torch.no_grad():
            for batch in val_bar:
                points = batch['points'].transpose(1, 2).to(device) 
                labels = batch['semantics'].to(device) 
                
                preds, trans_feat = model(points) 
                
                loss_seg = criterion(preds, labels)
                loss_reg = feature_transform_regularizer(trans_feat)
                loss = loss_seg + train_cfg['reg_weight'] * loss_reg
                
                val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Bilan Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # SAUVEGARDE
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'config': config 
        }

        # Sauvegarde systématique du dernier modèle
        last_filepath = save_dir / f"pointnet_last_{run_timestamp}.pth"
        torch.save(checkpoint, last_filepath)

        # Sauvegarde du meilleur modèle (basé sur la validation)
        if avg_val_loss < best_val_loss:
            print(f"Amélioration Val Loss est passée de {best_val_loss:.4f} à {avg_val_loss:.4f}.")
            print("Sauvegarde du nouveau meilleur modèle...")
            best_val_loss = avg_val_loss
            
            best_filepath = save_dir / "pointnet_best.pth"
            torch.save(checkpoint, best_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement de PointNet sur SemanticKITTI")
    parser.add_argument('--config_path', type=str, default='configs/pointnet_config.yaml', help='Chemin vers le fichier YAML')
    args = parser.parse_args()
    
    main(args)