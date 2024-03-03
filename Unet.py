import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#构造左边特征特区模块
class conv_block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch,out_ch,kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv1d(out_ch,out_ch,kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

#构造右边特征融合模块
class up_conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_ch,out_ch,kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x
#模型架构

class UNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNet,self).__init__()
        n1 = 32
        filters = [n1,n1*2,n1*4,n1*8]
        #最大池化层
        self.Maxpool1 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.Maxpool4 = nn.MaxPool1d(kernel_size=2,stride=2)
        #左边特征提取
        self.Conv1 = conv_block(in_ch,filters[0])
        self.Conv2 = conv_block(filters[0],filters[1])
        self.Conv3 = conv_block(filters[1],filters[2])
        self.Conv4 = conv_block(filters[2],filters[3])
        #self.Conv5 = conv_block(filters[3],filters[4])


        #右边特征融合
        #self.Up5 = up_conv(filters[4], filters[3])
        #self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv1d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        #e5 = self.Maxpool4(e4)
       # e5 = self.Conv5(e5)
        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        #d5 = self.Up5(e5)
        #d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接
        #d5 = self.Up_conv5(d5)
        #d4 = self.Up4(d5)
        #d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        #d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out



def train(model,train_loader,optimizer,criterion,device,epoch):
    iteration_loss_list = []
    model.to(device)

    for e in range(epoch):
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
            print("epoch: {} [{}/{} {:.2f}%] train loss: {} ".format(e, batch_idx * len(data),
                                                                                   len(train_loader.dataset),
                                                                                   100 * batch_idx/ len(train_loader),
                                                                                   loss.item())
                  )
        average_loss = running_loss / len(train_loader)
        print(f"Epoch [{e+1}/{epoch}], Train Loss: {average_loss:.4f}")


    return  iteration_loss_list,average_loss

def test(model,test_loader,criterion,device):
    model.eval()
    iteration_loss_list= []
    running_loss = 0
    with torch.no_grad(): #禁用梯度
        for batch_idx,(data,labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs,labels)
            #损失计算
            iteration_loss_list.append(float(loss))
            running_loss += loss.item()
    average_loss = running_loss / len(test_loader)
    print(f"Test Loss: {average_loss:.7f}")
    return iteration_loss_list,average_loss



if __name__ == "__main__":

    Gpu = torch.device("cuda")
    Cpu = torch.device("cpu")
    filter_net = UNet(in_ch=1,out_ch=1).to(Gpu)  # 模型加载到GPU上
    print(filter_net)                 # torch的print(model)只能打印层的细节，不包括每层的输出维度 有点遗憾
    summary(filter_net, (1,1024))  # summary()很像keras的model.summary()
    # x = torch.randn((2, 2000))
    # y = filter_net(x)
    # print(y.shape)


