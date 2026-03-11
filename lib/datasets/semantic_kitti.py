import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class SemanticKittiDataset(Dataset):
    """
    Dataset PyTorch pour SemanticKITTI.
    """
    def __init__(self, root_dir, sequences, transform=None, load_labels=True):
        self.root_dir = Path(root_dir)
        self.sequences = sequences
        self.transform = transform
        self.load_labels = load_labels
        
        self.scan_files = []
        self.label_files = []
        
        for seq in self.sequences:
            seq_dir = self.root_dir / "sequences" / seq
            
            scans = sorted(list((seq_dir / "velodyne").glob("*.bin")))
            self.scan_files.extend(scans)
            
            if self.load_labels:
                labels = sorted(list((seq_dir / "labels").glob("*.label")))
                assert len(scans) == len(labels), f"Différence scan/label dans la séquence {seq}"
                self.label_files.extend(labels)

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        scan_path = self.scan_files[idx]
        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        
        points = scan[:, :3] 
        remissions = scan[:, 3:]

        sample = {'points': points, 'remissions': remissions}

        if self.load_labels:
            label_path = self.label_files[idx]
            label_data = np.fromfile(label_path, dtype=np.uint32)
            
            semantics = label_data & 0xFFFF
            instances = label_data >> 16
            
            sample['semantics'] = semantics
            sample['instances'] = instances

        if self.transform:
            sample = self.transform(sample)

        return sample