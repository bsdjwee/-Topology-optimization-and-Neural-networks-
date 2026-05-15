import os
import torch
from torch.utils.data import Dataset

class TODiffusionDataset(Dataset):
    def __init__(self, folder, normalize=True):
        self.files = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.endswith(".pt")
        ]
        
        print(f"Found {len(self.files)} data files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file_path = self.files[idx]
            sample = torch.load(file_path, map_location='cpu')
    
            #density = sample["structure"] > 0.5).float().permute(0,2,1)  # 0/1 ºÚ°×Í¼
            density = sample["structure"] .float()
            density = torch.clamp(density, 0.0, 1.0)

            bc = sample["bc"].float()
            load = sample["load"].float()
            volfrac = sample["volfrac"].float()
    
            return {
                "density": density,
                "bc": bc,
                "load": load,
                "volfrac": volfrac
            }
    
        except Exception as e:
            print(f"[Dataset error] idx={idx}, file={self.files[idx]}")
            print(f"  -> {e}")
    
            raise

    
    def denormalize(self, density):
        if self.normalize:
            return density * self.density_std + self.density_mean
        return density