import torch
import torch.nn as nn
import os
import numpy as np

from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

for root, dirs, files in os.walk("Dataset"):
    for file in files:
        filepath = os.path.join(root, file)
        if 'seg' in file:
            continue
        img = np.load(filepath)
        plt.figure(figsize=(10,5))
        for i in range(3):
            plt.subplot(1, 4, i+1)
            plt.imshow(img[:,:,i], cmap='gray')
        seg = np.load(filepath.replace("img", "seg"))
        
        plt.subplot(1, 4, 4)
        plt.imshow(seg, cmap='gray')
        print(img.shape, seg.shape)
        plt.show()
        tens = ToTensor()(img)
        print(tens.shape)
        plt.cla()
        
            
            