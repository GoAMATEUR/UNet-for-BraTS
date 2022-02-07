"""
    By:     hsy
    Date:   2022/1/27
"""
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
        label = np.load(label_path).astype(np.float32)

        return transform(img), transform(label) # (3, 240, 240) tensor, (240, 240) tensor
    
    def get_img_size(self):
        H, W, C = np.load(os.path.join(self.data_root, self.images[0])).shape
        return H, W
        

if __name__ == "__main__":
    a = np.array([1,2])
    print(a.dtype)

