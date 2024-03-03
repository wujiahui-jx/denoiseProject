import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np

# 定义Unet网络的编码器部分
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

# 定义Unet网络的解码器部分
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

# 定义完整的Unet网络
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = UNetEncoder(in_channels, 64)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)
        self.decoder1 = UNetDecoder(512, 256)
        self.decoder2 = UNetDecoder(256, 128)
        self.decoder3 = UNetDecoder(128, 64)
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        skip1 = self.encoder1(x)#32,64,4096
        skip11 = self.Maxpool1(skip1)#32,64,2048
        skip2 = self.encoder2(skip11)#32,128,2048
        skip21 = self.Maxpool2(skip2)#32,128,1024
        skip3 = self.encoder3(skip21)#32,256,1024
        skip31 = self.Maxpool3(skip3)#32,256,512
        bottleneck = self.encoder4(skip31)#32,512,512
        x = self.decoder1(bottleneck, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder3(x, skip1)
        x = self.final_conv(x)
        return x

# 创建Unet网络实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=1, out_channels=1).to(device)


# 定义训练参数
learning_rate = 0.001
num_epochs = 5
batch_size = 32

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 准备数据
# 假设已经准备好训练数据和标签，分别存储在train_data和train_labels中
train_data = np.load('F:\Wujiahui\deoniseProject\\train_data_CH4.npy')
train_labels = np.load('F:\Wujiahui\deoniseProject\\test_data_CH4.npy')
train_data = train_data[:,:1024]
train_labels = train_labels[:,:1024]

x_train_1 = torch.tensor(train_data, dtype=torch.float)
x_test_1 = torch.tensor(train_labels, dtype=torch.float)
num, seq = x_train_1.size()
x_train_1 = torch.reshape(x_train_1, (num, 1, seq))

x_test_1 = torch.reshape(x_test_1, (num, 1, seq))
print(x_test_1.size())
print('What happenned')
# 将数据包装成TensorDataset
train_dataset = TensorDataset(x_train_1, x_test_1)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 开始训练
iteration_loss_list=[]

model.to(device)
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # 将数据和标签送入设备
        data, labels = data.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(data)
        # 计算损失
        loss = criterion(outputs, labels)
        iteration_loss_list.append(float(loss))
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # 统计损失
        running_loss += loss.item()
        print("epoch: {} [{}/{} {:.2f}%] train loss: {}  ".format(epoch, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100 * batch_idx / len(train_loader),
                                                                               loss.item())
              )
    average_loss = running_loss / len(train_loader)
    print("Epoch {} Average loss: {:.6f}".format(epoch, average_loss))
"""保存模型"""
model_save_path = r"F:\Wujiahui\deoniseProject\model_unet.pt"
torch.save(model.state_dict(), model_save_path)
summary(model, (1, 4096))
plt.figure()
plt.plot(iteration_loss_list, label="train loss")
plt.show()