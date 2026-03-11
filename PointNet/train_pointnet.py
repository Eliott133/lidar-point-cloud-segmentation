import argparse # <-- Le module pour lire la ligne de commande
import yaml     # <-- Pour lire le fichier config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.semantic_kitti import SemanticKittiDataset
from datasets.transforms import SemanticKittiLabelRemap, ToTensor, RandomRotationZ, FixedPointSampler
from pointnet import PointNetSegmentation 

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d, requires_grad=True).to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def main(args):
    # Chargement de la configuration YAML
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    train_cfg = config['training']
    data_cfg = config['dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lancement de l'entraînement sur : {device}")
    print(f"Fichier de configuration : {args.config_path}")

    # Pipelines de donnée
    train_transforms = transforms.Compose([
        SemanticKittiLabelRemap(),
        FixedPointSampler(num_points=train_cfg['num_points']), 
        RandomRotationZ(probability=0.5),
        ToTensor()
    ])

    train_dataset = SemanticKittiDataset(
        root_dir=data_cfg['root_dir'],
        sequences=data_cfg['train_sequences'], 
        transform=train_transforms,
        load_labels=True
    )
    
    if len(train_dataset) == 0:
        raise RuntimeError("Le Dataset est vide ! Vérifiez le chemin 'root_dir' dans votre YAML et assurez-vous que les fichiers .bin sont bien présents.")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=4
    )

    # Initialisation du Modèle
    model = PointNetSegmentation(config_path=args.config_path).to(device)
    
    # Optimiseur et Loss
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Boucle d'entraînement
    for epoch in range(train_cfg['epochs']):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")
        
        for batch in progress_bar:
            points = batch['points'].transpose(1, 2).to(device) 
            labels = batch['semantics'].to(device) 
            
            optimizer.zero_grad()
            preds, trans_feat = model(points) 
            
            loss_seg = criterion(preds, labels)
            loss_reg = feature_transform_regularizer(trans_feat)
            
            # Utilisation du paramètre de régularisation issu du YAML
            loss = loss_seg + train_cfg['reg_weight'] * loss_reg
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"-> Fin de l'Epoch {epoch+1} | Loss moyenne : {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement de PointNet sur SemanticKITTI")
    
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='../configs/pointnet_config.yaml', 
        help='Chemin vers le fichier YAML de configuration'
    )
    
    # Lecture des arguments passés dans le terminal
    args = parser.parse_args()
    
    main(args)