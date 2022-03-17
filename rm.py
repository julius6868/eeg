from random import sample
from signal import signal
import torch
from torch import nn
from torch.nn import functional as F
import bluetooth
import numpy as np
from time import sleep
from data_loader import *
from data_loader import data_generator_np
from data_preparation import data_preparation
# from models.MobileNet import MobileNetV3_Small
from models.ResNet32 import ResNet, BasicBlock
import torch.optim as optim
import random


class hswish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        out = x*self.relu6(x+3)/6
        return out

class hsigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        out = self.relu6(x+3)/6
        return out

class SE(nn.Module):
    def __init__(self, in_channels, reduce=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduce, 1, bias=False),
            nn.BatchNorm2d(in_channels//reduce),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels // reduce, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            hsigmoid()
        )
    def forward(self, x):
        out = self.se(x)
        out = x*out
        return out

class Block(nn.Module):
    def __init__(self, kernel_size, in_channels, expand_size, out_channels, stride, se=False, nolinear='RE'):
        super().__init__()
        self.se = nn.Sequential()
        if se:
            self.se = SE(expand_size)
        if nolinear == 'RE':
            self.nolinear = nn.ReLU6(inplace=True)
        elif nolinear == 'HS':
            self.nolinear = hswish()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expand_size, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_size),
            self.nolinear,
            nn.Conv2d(expand_size, expand_size, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            self.se,
            self.nolinear,
            nn.Conv2d(expand_size, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.stride = stride

    def forward(self, x):
        out = self.block(x)
        if self.stride == 1:
            out += self.shortcut(x)
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )
        self.neck = nn.Sequential(
            Block(3, 16, 16, 16, 2, se=True),
            Block(3, 16, 72, 24, 2),
            Block(3, 24, 88, 24, 1),
            Block(5, 24, 96, 40, 2, se=True, nolinear='HS'),
            Block(5, 40, 240, 40, 1, se=True, nolinear='HS'),
            Block(5, 40, 240, 40, 1, se=True, nolinear='HS'),
            Block(5, 40, 120, 48, 1, se=True, nolinear='HS'),
            Block(5, 48, 144, 48, 1, se=True, nolinear='HS'),
            Block(5, 48, 288, 96, 2, se=True, nolinear='HS'),
            Block(5, 96, 576, 96, 1, se=True, nolinear='HS'),
            Block(5, 96, 576, 96, 1, se=True, nolinear='HS'),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            hswish()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            hswish()
        )
        self.conv4 = nn.Conv2d(1280, 4, 1, bias=False)

        # self.fc = nn.Sequential(
        #     nn.Conv2d(200, 1024, 1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     # hswish()
        # )
        self.fc = nn.Sequential(
            nn.Linear(120, 1024),
            nn.BatchNorm2d(6),
            hswish(),
            nn.Linear(1024, 1024),
            nn.BatchNorm2d(6),
            hswish()
        )

    def forward(self, x):
        x = x.unsqueeze(-2)
        # x = x.permute(0, 3, 1, 2)
        x = self.fc(x)
        # x = x.permute(0, 2, 3, 1)
        x = x.view(x.size()[0], x.size()[1], 32, 32)
        # print("1111", x.size())
        x = self.conv1(x)
        # print("2222", x.size())
        x = self.neck(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(1)
        return F.log_softmax(x, dim=-1)

net_path = "net_model.pkl"
net = torch.load(net_path, map_location=torch.device('cpu'))

# ======================================================================
# bluetoth receiving
ble_address = '98:D3:31:F9:6B:67'
port = 1

sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
try: 
    sock.connect((ble_address, port))
    print("Connection Successful!!!")
except:
    print("Connection Fail!!!")

signal_channel = [[] for i in range(6)]

data = []
samples = []
sample_size = 120
buffer_size = 30000 # 5 min
over_lapping = 0.5
window_size = sample_size * over_lapping
idx = 0

while True:    
#     get ble data
    ble_data = sock.recv(1024)
#     if data length bigger than 10

    if len(ble_data) >= 16:
#         make sure the ble data's start and end met the check symble
#         make sure the data is correct
        if chr(ble_data[0])=='s' and chr(ble_data[15]=='e'):

            for i in range(6):
                signal_channel[i].append(ble_data[i+8])

            data_len = np.shape(signal_channel)[1]

            if data_len >= sample_size and data_len <= buffer_size:
                if data_len % sample_size == 0 or data_len % sample_size == 60:
                    signal_channel_ = np.array(signal_channel)
                    'sample will be passed to the Net as input!!!!!'
                    sample = signal_channel_[:, int(idx * window_size): data_len]
                    sample = torch.tensor(torch.FloatTensor(sample))
                    sample = sample.unsqueeze(0)
                    # print('sample size:', np.shape(sample))
                    
                    result = net(sample)
                    label = torch.max(result.data, 1)
                    print(label)

                     #==============================================================
                    """
                    if collecting the data, use samples to store the data
                    """
                    # samples.append(sample)
                     #==============================================================

                    idx += 1
            elif data_len > buffer_size:
                signal_channel = [[] for j in range(6)]
                idx = 0
                print('reset signal-channel')

                #==============================================================
                """
                if collecting the data, set the buffer size to 6000!!!!
                """
                # samples_ =np.array(samples)
                # samples_ = samples_.reshape(-1, 6 * 200)
                # np.savetxt('1', samples_)
                # break
                 #==============================================================
                   
            #         X = torch.tensor(data)
            #         X = X.unsqueeze(0)
            #         X = X.type(torch.FloatTensor)
            #         signal_channel = [[] for i in range(6)]
            #         result = net(X)
            #         label = torch.max(result.data, 1)
            #         print('result:', label)
                
            # if np.shape(signal_channel)[1] == 200: 
            #     data = np.array(signal_channel)
            #     X = torch.tensor(data)
            #     X = X.unsqueeze(0)
            #     X = X.type(torch.FloatTensor)
            #     signal_channel = [[] for i in range(6)]
            #     result = net(X)
            #     label = torch.max(result.data, 1)
            #     print('result:', label)
            #     # print(result)

