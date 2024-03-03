# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pywt
# Combined Model for Denoising and Regression

class WaveletTransformLayer(nn.Module):
    def __init__(self, wavelet_name='db4', mode='zero', input_length=1116):
        super(WaveletTransformLayer, self).__init__()

        self.wavelet_name = wavelet_name
        self.mode = mode
        self.input_length = input_length
        self.raw_threshold_value = nn.Parameter(torch.tensor(0.02))
        self.fc1 = nn.Linear(1116,100)
        self.fc2 = nn.Linear(100,1)
        #self.approx_coeff_length = pywt.dwt_coeff_len(self.input_length, filter_len=pywt.Wavelet(wavelet_name).dec_len,mode='symmetric')

    def forward(self, x):
        threshold_value_clamp = torch.clamp(self.raw_threshold_value,min=0.01)
        # 小波变换
        coeffs = pywt.wavedec(x.detach().cpu().numpy(), self.wavelet_name, mode=self.mode, level= 7)
        # 调整近似系数的大小

        coeffs_thresholded = [pywt.threshold(coeff, value=threshold_value_clamp.item(), mode='soft') for coeff in coeffs]
        reconstructed_signal = pywt.waverec(coeffs_thresholded, self.wavelet_name, mode=self.mode)
        reconstructed_signal = torch.tensor(reconstructed_signal, requires_grad=True).float().to(x.device)
        x = torch.sigmoid(self.fc1(reconstructed_signal))
        reg_signal = self.fc2(x)
        #approx_coeff = coeffs[0]
        return reconstructed_signal,reg_signal


class Model(nn.Module):     # 三层 hidden layer  2048-500-100-2048

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=1116, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=1000)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=1000, out_features=1116)
        # Principal component extration(as a linear layer)
        self.pca_layer = nn.Linear(in_features=1116, out_features=50)
        # Regression model
        self.regression_layer = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x_denoise = torch.sigmoid(x)
        x_pca = self.pca_layer(x_denoise)
        x_reg = self.regression_layer(x_pca)
        return x_denoise,x_reg

class CombinedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_components, output_dim):
        super(CombinedModel, self).__init__()

        # Denoising model
        self.denoise_layer1 = nn.Linear(input_dim, hidden_dim)
        self.denoise_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)

        self.denoise_layer3 = nn.Linear(hidden_dim, input_dim)

        # Principal component extration(as a linear layer)
        self.pca_layer = nn.Linear(input_dim, num_components)
        # Regression model
        self.regression_layer = nn.Linear(num_components, output_dim)

    def forward(self, x):
        # Denoising
        x_denoise = torch.relu(self.denoise_layer1(x))
        x_denoise = torch.relu(self.denoise_layer2(x_denoise))
        x_denoise = self.denoise_layer3(x_denoise)
        # Principal component extration
        x_pca = self.pca_layer(x_denoise)

        # Regression
        x_regression = self.regression_layer(x_pca)

        return x_denoise, x_regression


# 定义Unet网络的解码器部分
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.relu(x)

        return x

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

class CombineUNet(nn.Module):
    def __init__(self, in_channels, out_channels,input_dim,num_components,output_dim):
        super(CombineUNet, self).__init__()
        self.encoder1 = UNetEncoder(in_channels, 32)
        self.encoder2 = UNetEncoder(16, 32)
        self.encoder3 = UNetEncoder(32, 64)
        self.encoder4 = UNetEncoder(64, 128)
        self.decoder1 = UNetDecoder(128, 64)
        self.decoder2 = UNetDecoder(64, 32)
        self.decoder3 = UNetDecoder(32, 16)
        self.final_conv = nn.Conv1d(16, out_channels, kernel_size=1)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pca_layer = nn.Linear(input_dim, num_components)
        # Regression model
        self.regression_layer = nn.Linear(num_components, output_dim)
    def forward(self, x):
        skip1 = self.encoder1(x)#32,16,4096
        skip11 = self.Maxpool1(skip1)#32,16,2048
        skip2 = self.encoder2(skip11)#32,32,2048
        skip21 = self.Maxpool2(skip2)#32,32,1024
        skip3 = self.encoder3(skip21)#32,64,1024
        skip31 = self.Maxpool3(skip3)#32,64,512
        bottleneck = self.encoder4(skip31)#32,128,512
        x = self.decoder1(bottleneck, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder3(x, skip1)
        x_denoise = self.final_conv(x)
        x_pca = self.pca_layer(x_denoise)
        x_regression = self.regression_layer(x_pca)
        return x_denoise,x_regression

class CombineCNN(nn.Module):
    def __init__(self,input_dim, num_components,output_dim):
        super(CombineCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 128, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.pca_layer = nn.Linear(input_dim, num_components)
        self.regression_layer = nn.Linear(num_components, output_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.relu(self.fc1(x))
        x_denoise = self.sigmoid(self.fc2(x))
        x_pca = self.pca_layer(x_denoise)
        x_reg = self.regression_layer(x_pca)
        return x_denoise,x_reg



class Denoising1DCNN(nn.Module):
    def __init__(self):
        super(Denoising1DCNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),  # [batch, 16, 1116]
            nn.ReLU(),
            nn.MaxPool1d(2),  # [batch, 16, 558]
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),  # [batch, 32, 558]
            nn.ReLU(),
            nn.MaxPool1d(2),  # [batch, 32, 279]
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # [batch, 64, 279]
            nn.ReLU(),
            nn.MaxPool1d(2)  # [batch, 64, 139]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),  # [batch, 64, 139]
            nn.ReLU(),
            nn.Upsample(279),  # [batch, 64, 279]
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),  # [batch, 32, 279]
            nn.ReLU(),
            nn.Upsample(558),  # [batch, 32, 558]
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),  # [batch, 16, 558]
            nn.ReLU(),
            nn.Upsample(1116),  # [batch, 16, 1116]
            nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1)  # [batch, 1, 1116]
        )
        self.pca = nn.Linear(1116,50)
        self.reg = nn.Linear(50,1)
    def forward(self, x):
        x = self.encoder(x)
        x_denoise = self.decoder(x)
        x_pca = self.pca(x_denoise)
        x_reg = self.reg(x_pca)
        return x_denoise,x_reg


# 创建自定义的数据集
class MultiOutputDataset(Dataset):
    def __init__(self, X, y1, y2):
        self.X = X
        self.y1 = y1
        self.y2 = y2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], (self.y1[idx], self.y2[idx])



# Training loop
def train(model, train_loader, optimizer, criterion, num_epochs,alpha,device):
    model.train()
    model.to(device)
    denoiseloss_list = []
    regloss_list = []
    total_batch = 0
    total_denoiseloss = 0
    total_regloss = 0
    for epoch in range(num_epochs):
        # Forward pass
        for index,data in enumerate(train_loader):
            x_batch, denoise_target, reg_target = data
            x_batch = x_batch.contiguous()
            denoise_target = denoise_target.contiguous()
            x_batch,denoise_target,reg_target = x_batch.to(device), denoise_target.to(device), reg_target.to(device)
            x_denoised, y_regression = model(x_batch)
            optimizer.zero_grad()
            # Compute losses
            loss_denoise = criterion(x_denoised, denoise_target)
            loss_regression = criterion(y_regression, reg_target)
            denoiseloss_list.append(float(loss_denoise))
            regloss_list.append(float(loss_regression))
            # Combined loss
            combined_loss = alpha * loss_denoise + (1 - alpha) * loss_regression
            total_denoiseloss += loss_denoise.item()
            total_regloss += loss_regression.item()
            # Backward pass and optimization
            total_batch += 1
            combined_loss.backward()
            optimizer.step()
            print("epoch: {} train loss: {}  test loss: {}".format(epoch,loss_denoise.item(), loss_regression.item()))

        avg_denoiseloss = total_denoiseloss / total_batch
        avg_regloss = total_regloss / total_batch
        #print(f"Epoch [{epoch + 1}/{num_epochs}], Denoise Loss: {avg_denoiseloss:.6f}, Regression Loss: {avg_regloss:.6f}")
    return denoiseloss_list, regloss_list
'''
def train_frozen(model, train_loader, optimizer, criterion, epoch_denoise,epoch_regression,alpha,device):
    for epoch in range(epoch_denoise):
        for batch_data in train_loader:
            inputs, denoise_targets, _ = batch_data
            optimizer.zero_grad()
            denoised, _ = model(inputs)
            loss = criterion(denoised, denoise_targets)
            loss.backward()
            optimizer.step()
    for param in model.
'''


def eval(model, test_loader, criterion,device):
    model.eval()
    model.to(device)
    reg_list = []
    relative_error = []
    relative = 0
    with torch.no_grad():
        for index,data in enumerate(test_loader):
            x_batch, denoise_target, reg_target = data
            x_batch,denoise_target,reg_target = x_batch.to(device), denoise_target.to(device), reg_target.to(device)
            preds_denoise, preds_regression = model(x_batch)
            denoise_loss = criterion(preds_denoise, denoise_target)
            reg_lost = criterion(preds_regression, reg_target)
            reg_list.append(float(preds_regression))
            relative = abs(preds_regression - reg_target) / reg_target
            relative_error.append(float(relative))
    return reg_list, relative_error


'''def k_fold_cross_validation(model, dataset, criterion, optimizer, device, k=10, num_epochs=10, batch_size=4):
    kf = KFold(n_splits=k, shuffle=True)
    train_loss = 0
    test_loss = 0
    train_list = []
    test_list = []
    for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            train_loss = kfold_train(model, train_loader, criterion, optimizer, device)
            test_loss = kfold_eval(model, test_loader, criterion, device)
            train_list.append(float(train_loss))
            test_list.append(float(test_loss))
            print(f"Fold [{fold+1}/{k}], Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    return train_list,test_list
'''
