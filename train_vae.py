import logging
import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn

import iecdt_lab

model = CNNVAE(latent_dim=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(images)
        loss = loss_function(x_recon, images, mu, logvar)
        loss.backward()
        optimizer.step()