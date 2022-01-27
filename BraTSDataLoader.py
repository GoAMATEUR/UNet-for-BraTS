import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

class BraTSDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.files = os.listdir(data_root)
        self.images = self.files[::2] if "img" in self.files[0] else self.files[1::2]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int):
        file_path = os.path.join(self.data_root, self.images[index])
        label_path = os.path.join(self.data_root, self.images[index].replace("img", "seg"))
        img = np.load(file_path)
        label = np.load(label_path)
        return transform(img), transform(label) # (3, 240, 240) tensor, (240, 240) tensor

if __name__ == "__main__":
    path = "./dataset"
    files = os.listdir(path)
    print(files[::2][1145])
    print(files[1::2][1145])

