import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class TNet(nn.Module):
    """
    Réseau d'alignement spatial (Transformation Network).
    Il prédit une matrice de transformation affine (k x k) appliquée aux données 
    pour les rendre invariantes aux transformations géométriques (rotation, translation).
    """
    def __init__(self, k):
        """
        Args:
            k (int): Dimension de l'espace à transformer (ex: 3 pour XYZ, ou 64 pour les features).
        """
        super(TNet, self).__init__()
        self.k = k
        
        # Extraction de caractéristiques partagées
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Réseau entièrement connecté pour déduire la matrice
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialisation de la dernière couche pour qu'elle produise une matrice identité par défaut.
        # C'est crucial pour ne pas détruire les données dès la première itération.
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).flatten())

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (B, k, N)
        Returns:
            torch.Tensor: Matrice de transformation de forme (B, k, k)
        """
        B = x.size(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max Pooling Global pour obtenir une signature globale du nuage
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Remodelage en matrice (B, k, k)
        matrix = x.view(-1, self.k, self.k)
        return matrix


class PointNetSegmentation(nn.Module):
    """
    Architecture PointNet pour la segmentation sémantique de nuages de points.
    Générée dynamiquement via un fichier de configuration YAML, incluant les T-Nets.
    """
    def __init__(self, config_path):
        """
        Args:
            config_path (str): Chemin vers le fichier YAML contenant les hyperparamètres.
        """
        super(PointNetSegmentation, self).__init__()
        
        # --- Chargement de la configuration ---
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['model']
            
        self.in_channels = config['in_channels']
        self.num_classes = config['num_classes']
        self.local_feat_idx = config['encoder']['local_feature_index']
        self.feature_transform_idx = config['encoder'].get('feature_transform_index', 1) # Par défaut, après la 2ème conv
        self.dropout_rate = config.get('decoder', {}).get('dropout_rate', 0.0)
        
        # --- T-Nets ---
        # T-Net d'entrée (applique une transformation sur les coordonnées initiales)
        self.tnet_in = TNet(k=self.in_channels)
        
        # T-Net de caractéristiques (applique une transformation dans l'espace latent)
        enc_channels = config['encoder']['channels']
        feat_dim = enc_channels[self.feature_transform_idx]
        self.tnet_feat = TNet(k=feat_dim)

        # --- Construction dynamique de l'Encodeur ---
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        
        in_ch = self.in_channels
        for out_ch in enc_channels:
            self.encoder_convs.append(nn.Conv1d(in_ch, out_ch, 1))
            self.encoder_bns.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch 
            
        # --- Calcul de la dimension d'entrée du Décodeur ---
        local_dim = enc_channels[self.local_feat_idx]
        global_dim = enc_channels[-1]
        decoder_in_ch = local_dim + global_dim
        
        # --- Construction dynamique du Décodeur ---
        dec_channels = config['decoder']['channels']
        self.decoder_convs = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()
        
        in_ch = decoder_in_ch
        for out_ch in dec_channels:
            self.decoder_convs.append(nn.Conv1d(in_ch, out_ch, 1))
            self.decoder_bns.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
            
        # --- Couche finale ---
        self.final_conv = nn.Conv1d(in_ch, self.num_classes, 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        """
        Passe avant du modèle.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (B, in_channels, N)
                              où B est la taille du batch, et N le nombre de points.
                              
        Returns:
            torch.Tensor: Tenseur de sortie de forme (B, num_classes, N)
                          contenant les logits pour chaque point.
            (Optionnel) trans_feat: On pourrait retourner la matrice pour régulariser la loss.
        """
        B, C, N = x.size()
        local_features = None

        # --- T-Net d'Entrée ---
        trans_in = self.tnet_in(x) # Forme: (B, C, C)
        # Multiplication matricielle en lot : (B, C, C) x (B, C, N) -> (B, C, N)
        x = torch.bmm(trans_in, x)

        # --- Passage dans l'Encodeur ---
        for i in range(len(self.encoder_convs)):
            x = F.relu(self.encoder_bns[i](self.encoder_convs[i](x)))
            
            # Application du T-Net de caractéristiques (Feature Transform)
            if i == self.feature_transform_idx:
                trans_feat = self.tnet_feat(x) # Forme: (B, feat_dim, feat_dim)
                x = torch.bmm(trans_feat, x)   # Forme: (B, feat_dim, N)
                
            # Sauvegarde des features locales pour la segmentation
            if i == self.local_feat_idx:
                local_features = x

        # --- Max Pooling Global ---
        global_feature = torch.max(x, 2, keepdim=True)[0] # Forme: (B, global_dim, 1)

        # --- Préparation pour le Décodeur ---
        # Répéter le vecteur global N fois
        global_feature_expanded = global_feature.repeat(1, 1, N) # Forme: (B, global_dim, N)
        # Concaténation des informations locales (détail) et globales (contexte)
        concat_features = torch.cat([local_features, global_feature_expanded], dim=1) # Forme: (B, local_dim + global_dim, N)

        # --- Passage dans le Décodeur ---
        out = concat_features
        for i in range(len(self.decoder_convs)):
            out = F.relu(self.decoder_bns[i](self.decoder_convs[i](out)))
            # Dropout appliqué avant la dernière étape d'extraction
            if i == len(self.decoder_convs) - 1:
                out = self.dropout(out)

        # --- Classification point par point ---
        out = self.final_conv(out) # Forme: (B, num_classes, N)

        return out, trans_feat # On retourne aussi la transformation pour la régularisation de la loss  