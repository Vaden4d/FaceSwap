import numpy as np
import argparse
import os
import json

import torch
from torch.optim import Adam
from utils import loss_fn, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from Model import VAE
from Trainer import Trainer
from Data import Data, Splitter
from torch.utils.data import DataLoader

from catalyst.dl.experiments import SupervisedRunner

parser = argparse.ArgumentParser()
parser.add_argument('--trainimages', type=str, default='../only_faces_one')
#parser.add_argument('--masksdir', type=str, default='data/train/masks/')
#parser.add_argument('--testimages', type=str, default='data/test/images')
parser.add_argument('--logdir', type=str, default='data/logs')
parser.add_argument('--chkpdir', type=str, default='data/chkp')
parser.add_argument('--chkpname', type=str, default='none')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--train_batch_size', type=int, default=50)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip_norm', type=float, default=0.1)
parser.add_argument('--pretrained', type=str, default='None')
parser.add_argument('--output_weights', type=str, default='vae_output.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating dataloaders
dataset = Data('../only_faces_one')
splitter = Splitter(dataset)
train_dataset, test_dataset = splitter.train_test_split(test_size=0.2)
del dataset

train_dataset.batch_size = args.train_batch_size
test_dataset.batch_size = args.test_batch_size

train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

# optimizer
lr = args.lr
loaders = {'train': train_loader, 'valid': test_loader}
model = VAE(device).to(device)
if args.pretrained != 'None':
    model.load_state_dict(torch.load(args.pretrained))
optimizer = Adam(model.parameters(), lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, eps=1e-4)
clip_norm = args.clip_norm
criterion = loss_fn
num_epochs = args.num_epochs
logdir = './logdir'

for epoch in range(args.num_epochs):
    for idx, images in enumerate(train_loader):

        recon_images, mu, logvar = model(images.to(device))
        #print(recon_images, mu, logvar)
        loss, bce, kld = loss_fn(recon_images, images.to(device), mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                args.num_epochs,
                                loss.data/args.train_batch_size,
                                bce.data/args.train_batch_size,
                                kld.data/args.train_batch_size)
        print(to_print)
    torch.save(model.state_dict(), args.output_weights)

'''# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)'''

#writer = SummaryWriter(args.logdir)
#trainer = Trainer(model, optimizer, criterion, metric, clip_norm, writer, device)
