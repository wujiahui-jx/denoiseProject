import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import pickle
import torch.utils.data as Data
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import pywt
import torch
# Attention module
class WaveletActivation(nn.Module):
    def __init__(self):
        super(WaveletActivation, self).__init__()

    def forward(self, x):
        # Apply Morlet wavelet function
        return torch.cos(1.75 * x) * torch.exp(-(x**2))
class AttentionModule(nn.Module):
    def __init__(self, input_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        attention_weights = self.attention(x)
        x_weighted = x * attention_weights
        return x_weighted


class DenoiseModel(nn.Module):  # 三层 hidden layer  2048-500-100-2048

    def __init__(self):
        super(DenoiseModel, self).__init__()
        self.fc1 = nn.Linear(in_features=1116, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=1000)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(in_features=1000, out_features=1116)
        self.wavelet = WaveletActivation()
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = self.dropout1(x)
        x = torch.sigmoid(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


class AttentionDenoisingModel(nn.Module):
    def __init__(self, input_size):
        super(AttentionDenoisingModel, self).__init__()
        self.attention_module = AttentionModule(input_size)
        self.denoising_module = DenoiseModel()

    def forward(self, x):
        x_weighted = self.attention_module(x)
        denoised_output = self.denoising_module(x_weighted)
        return denoised_output


class WaveletTransformLayer(nn.Module):
    def __init__(self, wavelet_name='db4', mode='zero', input_length=1116):
        super(WaveletTransformLayer, self).__init__()
        self.wavelet_name = wavelet_name
        self.mode = mode
        self.input_length = input_length
        self.raw_threshold_value = nn.Parameter(torch.tensor(0.02))
        #self.approx_coeff_length = pywt.dwt_coeff_len(self.input_length, filter_len=pywt.Wavelet(wavelet_name).dec_len,mode='symmetric')

    def forward(self, x):
        threshold_value_clamp = torch.clamp(self.raw_threshold_value,min=0.01)
        # 小波变换
        coeffs = pywt.wavedec(x.detach().cpu().numpy(), self.wavelet_name, mode=self.mode, level= 7)
        # 调整近似系数的大小

        coeffs_thresholded = [pywt.threshold(coeff, value=threshold_value_clamp.item(), mode='soft') for coeff in coeffs]
        reconstructed_signal = pywt.waverec(coeffs_thresholded, self.wavelet_name, mode=self.mode)
        #approx_coeff = coeffs[0]
        return torch.tensor(reconstructed_signal, requires_grad=True).float().to(x.device)
'''
class WaveletTransformLayer1(nn.Module):
    def __init__(self, wavelet_name='db4', mode='zero', input_length=1116):
        super(WaveletTransformLayer1, self).__init__()
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
'''
class InvWaveletTransformLayer(nn.Module):
    def __init__(self, wavelet_name='db4', mode='zero'):
        super(InvWaveletTransformLayer, self).__init__()
        self.wavelet_name = wavelet_name
        self.mode = mode

    def forward(self, x):
        # 根据小波系数重建信号
        reconstructed_signal = pywt.waverec([x.detach().cpu().numpy()], self.wavelet_name, mode=self.mode)
        return torch.tensor(reconstructed_signal, requires_grad=True).float().to(x.device)



class WaveletDenoisingNet(nn.Module):
    def __init__(self, input_dim):
        super(WaveletDenoisingNet, self).__init__()
        self.wavelet_transform = WaveletTransformLayer(input_length=input_dim)
        # 根据小波变换后的输出维度调整FC层的输入大小
        self.fc1 = nn.Linear(1116, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, 300)
        self.fc3 = nn.Linear(300,1116)
        self.inv_wavelet_transform = InvWaveletTransformLayer()

    def forward(self, x):
        x = self.wavelet_transform(x)
        #x = self.inv_wavelet_transform(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def array_to_tensor(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.type(torch.cuda.FloatTensor)
    return tensor


def ConvBNReLU(in_channels, out_channels, kernel_size, stride):
    """
    自定义包括BN ReLU的卷积层，输入(N,in_channels,in_Length)
    输出(N, out_channels, out_Length)，卷积后进行批归一化，
    然后进行RELU激活。
    :param in_channels: 输入张量的通道数
    :param out_channels: 输出张量的通道数
    :param kernel_size: 卷积核尺寸
    :param stride: 卷积核滑动步长
    :return: BN RELU后的卷积输出
    """
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


def pooling1d(input_tensor, kernel_size=3, stride=2):
    """
    最大池化函数，输入形如(N, Channels, Length),
    输出形如(N, Channels, Length)的功能。
    :param input_tensor:要被最大池化的输入张量
    :param kernel_size:池化尺寸。在多大尺寸内进行最大池化操作
    :param stride:池化层滑动补偿
    :return:池化后输出张量
    """
    result = F.max_pool1d(input_tensor, kernel_size=kernel_size, stride=stride)
    return result


def average_pool1d(input_tensor, kernel_size=3, stride=2):
    result = F.avg_pool1d(input_tensor, kernel_size=kernel_size, stride=stride)
    return result


def global_average_pooling1d(input_tensor, output_size=1):
    """
    全局平均池化函数，将length压缩成output_size。
    输入(N, C, Input_size)
    输出(N, C, output_size)
    :param input_tensor: 输入张量
    :param output_size: 输出张量
    :return:全剧平均池化输出
    """
    result = F.adaptive_max_pool1d(input_tensor, output_size)
    return result

class Model(nn.Module):     # 三层 hidden layer  2048-500-100-2048

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=1116, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        self.fc3 = nn.Linear(in_features = 100,out_features= 1000)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(in_features=1000, out_features=1116)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x= self.fc3(x)
        x = self.dropout1(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

def read_pickle_to_array(path, name):
    """
    读取二进制文件并创建为np array类型
    :param path: 读取文件的路径
    :param name: 读取文件的文件名
    :return: ndarray类型的数据
    """
    file = open(path + name, "rb")
    array = pickle.load(file)
    array = np.array(array)
    return array


def Data_set(input_data, label_data, batch_size):
    """
    生成data_loader实例。可以定义batch_size
    :param input_data: 希望作为训练input的数据，tensor类型
    :param label_data: 希望作为训练label的数据，tensor类型
    :param batch_size: batch size
    :return: data_loader实例
    """
    data_set = Data.TensorDataset(input_data, label_data)
    data_loader = Data.DataLoader(dataset=data_set,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=0)
    return data_loader
def training(model, train_loader,test_x, test_y, device, optimizer, criterion, epochs):
    """
    训练模型，输入模型，数据集，GPU设备，选择的优化器以及损失函数，在设置的epoch内进行模型优化。
    adjust开启时将根据epoch自适应的调整learning rate。
    只有当adjust开启时，plot才能开启。否则plot功能永远关闭。
    plot开启时，将绘制输入的plot_ch4以及plot_label。每一代之后根据更新优化的参数，模型计算plot_ch4，并绘制输出与plot_label进行
    对比
    :param model: 输入的训练模型, untrained model
    :param train_loader: 输入的训练数据集
    :param test_x: using for compute test error
    :param optimizer: the chosen optimizer
    :param criterion: the loss function
    :param epochs: iteration running on models
    :param plot_ch4: a sample chosen from dataset to be computed and plotted
    :param plot_label: a sample label chosen from dataset to be plotted
    :param adjust: adaptive learning rate along with epochs when switch to True
    :param plot: only adjust = True will switch on.
    :return: trained loss & test loss
    """
    model.train()
    model.to(device)
    iteration_loss_list = []
    test_error_list = []
    for e in range(epochs):
        for index, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.contiguous()
            batch_y = batch_y.contiguous()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            optimizer.zero_grad()
            prediction1 = model(batch_x)
            prediction2 = model(test_x)
            loss = criterion(prediction1, batch_y)
            iteration_loss_list.append(float(loss))
            test_error = criterion(prediction2, test_y)
            test_error_list.append(float(test_error))
            loss.backward()
            optimizer.step()
            print("epoch: {} [{}/{} {:.2f}%] train loss: {}  test loss: {}".format(e, index * len(batch_x),
                                                                                   len(train_loader.dataset),
                                                                                   100 * index / len(train_loader),
                                                                                   loss.item(), test_error.item())
                  )

    return iteration_loss_list,test_error_list # , epoch_loss_list

if __name__ == "__main__":

    Gpu = torch.device("cuda")
    Cpu = torch.device("cpu")
    filter_net = Model().to(Gpu)  # 模型加载到GPU上
    print(filter_net)                 # torch的print(model)只能打印层的细节，不包括每层的输出维度 有点遗憾
    summary(filter_net, (1, 1116))  # summary()很像keras的model.summary()
    # x = torch.randn((2, 2000))
    # y = filter_net(x)
    # print(y.shape)

