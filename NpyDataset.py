from torch.utils.data import Dataset
import torch
import numpy as np
import os

class NpyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.file_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            files = [file for file in os.listdir(class_dir) if file.endswith('.npy')]
            for file in files:
                self.file_paths.append(os.path.join(class_dir, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        tensor = torch.from_numpy(np.load(file_path))  # Load .npy file and convert to tensor
        label = self.labels[idx]
        
        if len(tensor.shape) == 2:  # If the tensor is 2D, add a channel dimension
            tensor = tensor.unsqueeze(0)  # Adds a channel dimension at the beginning
        
        return tensor, label