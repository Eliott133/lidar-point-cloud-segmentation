import torch

from pointnet import PointNetSegmentation


model = PointNetSegmentation("../configs/config_pointnet.yaml")
print(model)

# Test avec un tenseur bidon : Batch=2, Channels=4, Points=1024
dummy_input = torch.randn(2, 4, 1024)
output = model(dummy_input)
print(f"Shape de sortie : {output.shape}") # Doit être [2, 20, 1024]