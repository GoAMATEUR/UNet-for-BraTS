"""
    By:     hsy
    Date:   2022/1/28
    TODO: eval Dice/IoU/GIoU
"""
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import os
from UNet import UNet
from BraTSDataLoader import BraTSDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./out_models/"
data_root = "./test/"
save_root = "./output/"

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
    save_image(output, os.path.join(save_root, f"{i}.png"))
