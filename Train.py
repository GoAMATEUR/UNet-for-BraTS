"""
    By:     Hsy
    Date:   2022/1/29
"""
import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from net.unet import UNet
from utils.dataloader import BraTSDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, help="Tag of this training session")
parser.add_argument("--backbone", type=str, help="type of backbone net, VGG16 or ResNet50", default="VGG16")
parser.add_argument("--batch_size", type=int, help="batch size", default=8)
parser.add_argument("--data_root", type=str, help="root of train set", default=".\\dataset")
parser.add_argument("--model_root", type=str, help="root of train set", default=".\\out_models")
parser.add_argument("--log_root", type=str, help="root of train set", default=".\\logs")
parser.add_argument("--model_path", type=str, help="(Optional) Path of pretrained model", default=None)
parser.add_argument("--lr", type=float, help="initial learning rate", default=1e-3)
args = parser.parse_args

train_tag = args.tag
net_type = args.backbone
batch_size = args.batch_size
data_root = args.data_root
model_root = os.path.join(args.model_root, train_tag)
log_root = os.path.join(args.log_root, train_tag)
model_path = args.model_path
learning_rate = args.lr
writer = SummaryWriter(log_root)
if not os.path.exists(model_root):
    os.mkdir(model_root)

# load trainset
dataset = BraTSDataset(data_root)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
W, H = dataset.size()

# Construct net
net = UNet(net_type).to(device) # to GPU

# print info
print(f"Traning Session {train_tag} \nBrief:\nBackbone_type:{net_type} | batch_size:{batch_size} | lr:{learning_rate} | model_root:{model_root} | Log root:{log_root}")
print(f"img_size: {W}*{H}")
print(f"Is GPU available: {torch.cuda.is_available()}")

# if using existing weights
if model_path is not None:
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print("{} model loaded.".format(model_path))
    else:
        raise("Model not found in {}.".format(model_path))

print("\nStart training ...")

# optimizer
opt = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(opt, mode="min", patience=4, verbose=True)

# define loss function
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
    
    
        