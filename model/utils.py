import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import time

#mean_faces = np.array([0.2937, 0.2937, 0.2912])
#std_faces = np.array([0.2257, 0.1962, 0.1906])


transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Lambda(lambda x: x / 255.0),
    #transforms.Normalize(mean_faces, std_faces)
])

'''
inverse_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(-mean_faces / std_faces, 1.0 / std_faces),
    transforms.Lambda(lambda x: x * 255.0)
])'''

def generate_batch(metadata, batch_size=3):
    n_batches = np.ceil(metadata.shape[0] / batch_size).astype(int)
    for i in range(n_batches):
        labels = metadata.iloc[i*(batch_size): (i+1)*batch_size].iloc[:, 1:].values
        names = metadata.iloc[i*(batch_size): (i+1)*batch_size].id
        yield download_data(names), labels

def download_data(data_path, files):

    images = []
    for file in files:
        img = Image.open(os.path.join(data_path, file))
        img = transform(img).numpy()
        images.append(img)

    images = np.array(images)
    images = torch.FloatTensor(images)

    return images

def loss_fn(recon_x, x, mu, logvar):

    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

if __name__ == '__main__':

    data_path = '../only_faces_one'
    _, _, files = next(os.walk(data_path))
    images = download_data(data_path, files)
    torch.save(images, 'images.pt')
    #tensor([0.2948, 0.2948, 0.2897])
    #tensor([0.2257, 0.1962, 0.1905])
