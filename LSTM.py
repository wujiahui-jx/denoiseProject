# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:55:31 2023

@author: 11527
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
def generate_data(num_samples):
    x = np.linspace(0, 5, num_samples)
    y = np.sin(x)
    noise = np.random.normal(0, 0.5, y.shape)
    noisy_y = y + noise
    return torch.tensor(noisy_y, dtype=torch.float32).unsqueeze(1)

noisy_data = generate_data(100)

# 定义生成器和鉴别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 初始化
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

# 训练
num_epochs = 1000
for epoch in range(num_epochs):
    # Train Discriminator
    optimizer_d.zero_grad()
    real_data = noisy_data
    real_labels = torch.ones((100, 1))
    fake_data = generator(real_data).detach()
    fake_labels = torch.zeros((100, 1))
    logits_real = discriminator(real_data)
    logits_fake = discriminator(fake_data)
    loss_real = criterion(logits_real, real_labels)
    loss_fake = criterion(logits_fake, fake_labels)
    loss_d = loss_real + loss_fake
    loss_d.backward()
    optimizer_d.step()

    # Train Generator
    optimizer_g.zero_grad()
    fake_data = generator(real_data)
    logits_fake = discriminator(fake_data)
    loss_g = criterion(logits_fake, real_labels)
    loss_g.backward()
    optimizer_g.step()

# 使用GAN去噪
with torch.no_grad():
    denoised_data = generator(noisy_data).numpy()

# 绘图
x = np.linspace(0, 5, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, noisy_data, 'o', label='Noisy Data', color='red')
plt.plot(x, denoised_data, 'o', label='Denoised Data', color='green')
plt.legend()
plt.show()
