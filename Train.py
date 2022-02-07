"""
    By:     Hsy
    Update: 2022/2/7
"""
import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from net.unet import UNet
from utils.dataloader import BraTSDataset
from utils.loss import BinaryDiceLoss, MetricsTracker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--tag",            type=str,   help="Tag of this training session")
parser.add_argument("--backbone",       type=str,   help="type of backbone net, VGG16 or ResNet50",     default="VGG16")
parser.add_argument("--batch_size",     type=int,   help="batch size",                                  default=8)
parser.add_argument("--summary_writer", type=int,   help="use summary writer as log",                   default=0)
parser.add_argument("--data_root",      type=str,   help="root of train set",                           default="./dataset")
parser.add_argument("--model_root",     type=str,   help="root of train set",                           default="./out_models")
parser.add_argument("--log_root",       type=str,   help="root of train set",                           default="./logs")
parser.add_argument("--model_path",     type=str,   help="(Optional) Path of pretrained model",         default=None)
parser.add_argument("--lr",             type=float, help="initial learning rate",                       default=1e-3)
args = parser.parse_args()

train_tag = args.tag
net_type = args.backbone
batch_size = args.batch_size
data_root = args.data_root
summary_writer = args.summary_writer
model_root = os.path.join(args.model_root, train_tag)
log_root = os.path.join(args.log_root, train_tag)
model_path = args.model_path
learning_rate = args.lr
if summary_writer == 1:
    writer = SummaryWriter(log_root)
if not os.path.exists(model_root):
    os.makedirs(model_root)
if not os.path.exists(log_root):
    os.makedirs(log_root)
# load trainset
dataset = BraTSDataset(data_root)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
H, W = dataset.get_img_size()

# Construct net
net = UNet(net_type, is_train=True).to(device) # to GPU

# print info
print("Traning Session {} \nBrief:\nBackbone_type:{} | batch_size:{} | lr:{} | model_root:{} | Log root:{}".format(train_tag, net_type, batch_size, learning_rate, model_root, log_root))
print("img_size:{}*{} | trainset size:{}".format(W, H, len(dataset)))
print("Is GPU available: {}".format(torch.cuda.is_available()))

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
loss_bce = nn.BCELoss()
loss_dice = BinaryDiceLoss()

# keep track of metrics
metrics = MetricsTracker(train_tag, log_root)

epoch = 1
weight_bce = 0.5
weight_dice = 0.5
while True:
    
    metrics.set_epoch(epoch)
    
    loss_sum = 0
    iter_count = 0
    min_loss = np.inf
    for i, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)
        out_image = net(img)
        
        iter_bce_loss = loss_bce(out_image, label)
        
        one_hot = (out_image>0.5).float()
        
        # dice loss + Binary Cross-Entropy loss
        iter_dice_loss = loss_dice(one_hot, label)
        
        iter_loss = weight_bce * iter_bce_loss + weight_dice * iter_dice_loss
        iter_loss.requires_grad_(True)
        
        opt.zero_grad()
        iter_loss.backward()
        opt.step()
        
        # cpu_img = img.detach().cpu()
        # cpu_out = out_image.detach().cpu()
        
        if i % 10 == 0:
            print("--Epoch: {}, iter: {}, iter_loss: {}".format(epoch, i, iter_loss.item()))
        
        with torch.no_grad():
            loss_sum += iter_loss.item()
            iter_count += 1
            if iter_loss.item() < min_loss:
                min_loss = iter_loss.item()
            metrics.update(one_hot, label, iter_dice_loss.item())
            # update metrics
    val_loss = loss_sum / iter_count
    acc, iou, dice = metrics.get_metrics()
    
    print("Epoch {} | val_loss:{} | min_loss:{} | Acc:{} | IoU:{} | Dice:{} ".format(epoch, val_loss, min_loss, acc, iou, dice))
    # save state-dict
    if ((epoch - 1) % 5 == 0):
        torch.save(net.state_dict(), os.path.join(model_root, "{}_{}.pth".format(train_tag, epoch)))
        metrics.save_logs()
        
    # tensorboard
    if summary_writer == 1:
        epoch_sample = [img[:,0,:,:].reshape(-1, 1, H, W), label, out_image]
        tags = ['img', 'label', 'out']
        for i in range(3):
            writer.add_images("epoch {}".format(epoch), epoch_sample[i])
        writer.add_scalar("{} Loss".format(train_tag), val_loss, epoch)
        writer.close()
    
    scheduler.step(val_loss)
    epoch += 1
    torch.cuda.empty_cache()
    
    
        