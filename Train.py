"""
    By:     hsy
    Date:   2022/1/28
"""
import os
import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from UNet import *
from BraTSDataLoader import BraTSDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: use argparse
train_tag = "train_1"
learning_rate = 1e-3
batch_size = 8
data_root = "./dataset"
model_root = f"./out_models/{train_tag}"
log_root = f"./logs/{train_tag}"
use_exist_model = False
model_path = None
writer = SummaryWriter(log_root)

if not os.path.exists(model_root):
    os.mkdir(model_root)

# load trainset
dataset = BraTSDataset(data_root)
W, H = dataset.size()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
net = UNet().to(device) # to GPU

print(f"Traning Session {train_tag} \nBrief:\nbatch_size:{batch_size} | lr:{learning_rate} | model_root:{model_root} | Log root:{log_root}")
print(f"img_size: {W}*{H}")
print(f"Is GPU available: {torch.cuda.is_available()}")

# if using existing weights
if use_exist_model:
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print("{} model loaded.".format(model_path))
    else:
        raise("Model not found in {}.".format(model_path))

print("\nStart training ...")
opt = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(opt, mode="min", patience=4, verbose=True)
loss_f = nn.BCELoss()
epoch = 1
while True:
    loss_sum = 0
    iter_count = 0
    min_loss = np.inf
    for i, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)
        out_image = net(img)
        iter_loss = loss_f(out_image, label)
        
        opt.zero_grad()
        iter_loss.backward()
        opt.step()
        
        if i % 100 == 0:
            print("--Epoch: {}, iter: {}, iter_loss: {}".format(epoch, i, iter_loss.item()))
        
        loss_sum += iter_loss.item()
        iter_count += 1
        if iter_loss.item() < min_loss:
            min_loss = iter_loss.item()
            
    avg_loss = loss_sum / iter_count
    print("Epoch {} | avg_loss:{} | min_loss: {}".format(epoch, avg_loss, min_loss))
    if ((epoch-1) % 5 == 0):
        torch.save(net.state_dict(), os.path.join(model_root, "{}_{}.pth".format(train_tag, epoch)))
    
    # tensorboard
    epoch_sample = [img[:,0,:,:].reshape(-1, 1, 240, 240), label,  out_image]
    tags = ['img', 'label', 'out']
    for i in range(3):
        writer.add_images("epoch {}".format(epoch), epoch_sample[i])
    writer.add_scalar("{} Loss".format(train_tag), avg_loss, epoch)
    writer.close()
    
    epoch += 1
    
    
        