"""
    By:     hsy
    Date:   2022/1/28
    TODO: eval Dice/IoU/GIoU
"""
from filecmp import cmp
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import os
from unet import UNet
from dataloader import BraTSDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./out_models/train_BCE_141.pth"
data_root = "./Val/Yes"
save_root = "./output/BCE_141/"

if not os.path.exists(model_path):
    raise("Model not fond")

if not os.path.exists(save_root):
    os.mkdir(save_root)

net = UNet().to(device)
net.load_state_dict(torch.load(model_path))
print(f"{model_path} loaded")

dataloader = DataLoader(BraTSDataset(data_root), batch_size=1, shuffle=False)
converter = ToPILImage()

for i, (img, label) in enumerate(dataloader):
    img, label = img.to(device), label.to(device)
    
    
    output = net(img)
    # a = img[0,0,:,:].detach().cpu().numpy()
    # plt.imshow(a, cmap='gray')

    plt.figure(figsize=(8,2.5))
    
    plt.subplot(1,5,1)
    plt.imshow(img[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    plt.title('layer1')  
    plt.xticks([]),plt.yticks([])  
    plt.subplot(1,5,2)
    plt.imshow(img[0,1,:,:].detach().cpu().numpy(), cmap='gray')
    plt.title('layer2')  
    plt.xticks([]),plt.yticks([])  
    plt.subplot(1,5,3)
    plt.imshow(img[0,2,:,:].detach().cpu().numpy(), cmap='gray')
    plt.title('layer3')  
    plt.xticks([]),plt.yticks([])  
    plt.subplot(1,5,4)
    plt.imshow(label[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    plt.title('label')  
    plt.xticks([]),plt.yticks([])  
    plt.subplot(1,5,5)
    plt.imshow(output[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    plt.title('output')  
    plt.xticks([]),plt.yticks([])  
    plt.savefig(os.path.join(save_root, f"{i}.jpg"))
    plt.close()
