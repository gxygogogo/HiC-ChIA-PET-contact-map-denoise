import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import re
import random

#%% params
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
epoch_size = 256

#%% 定义Res_block模块
class ResBlock(nn.Module):
    def __init__(self, feature_size = 32):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(feature_size, feature_size * 4, kernel_size = 1)
        self.conv2 = nn.Conv2d(feature_size * 4, feature_size, kernel_size = 1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size = 7, padding = 3)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        return out

#%% 定义卷积网络块
class ConvModel(nn.Module):
    def __init__(self, feature_size = 32):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, feature_size, kernel_size = 7)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size = 5)
        self.res_block = ResBlock(feature_size)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size = 5)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size = 3)
        self.conv5 = nn.Conv2d(feature_size, feature_size, kernel_size = 5)
        self.conv6 = nn.Conv2d(feature_size, feature_size, kernel_size = 3)
        self.conv7 = nn.Conv2d(feature_size, feature_size, kernel_size = 5)
        self.conv8 = nn.Conv2d(feature_size, 1, kernel_size = 5)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.res_block(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        return out

#%% 定义进行训练的神经网络模型
def model(train_input_dir, valid_input_dir, saver_dir, feature_size = 32, iterations_size = 10,):
    ## 实例化神经网络模型
    Conv_model = ConvModel(feature_size)
    res_block = Conv_model.res_block
    ## low-resolution
    input_x = torch.empty((None, 1, 40, 40), dtype = torch.float32)
    ## high-resolution
    input_y = torch.empty((None, 1, 28, 28), dtype = torch.float32)
    output_x = Conv_model(input_x)

    ## 使用MSE计算损失函数
    MSE_loss = nn.MSELoss()
    loss = MSE_loss(output_x, input_y)
    # loss = F.mse_loss(output_x, input_y)

    ## 损失函数和优化器
    optimizer = torch.optim.Adam(res_block.parameters(), lr = 0.001)

    print("Begin training...")
    try:
        for epoch in range(iterations_size):
            for file in os.listdir(train_input_dir):
                x = np.load(os.path.join(train_input_dir, file)).astype(np.float32)
                x = np.reshape(x, [x.shape[0], 40, 68, 1])
                size_input = int(x.shape[0] / epoch_size) + 1
                np.random.shuffle(x)
                total_loss = 0  # Total loss per iteration

                for i in range(size_input):
                    if i * epoch_size + epoch_size <= x.shape[0]:
                        input_data = torch.tensor(x[i * epoch_size:i * epoch_size + epoch_size, :, 0:40])
                        truth = torch.tensor(x[i * epoch_size:i * epoch_size + epoch_size, 0:28, 40:68])

                        optimizer.zero_grad()
                        output = res_block(input_data)
                        loss = MSE_loss(output, truth)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        if i % 10 == 1:
                            print("in the %sth iteration %sth step, the training loss is %f" % (epoch, i, loss.item()))

                        if i % 50 == 1 and epoch >= 10:
                            torch.save(res_block.state_dict(), os.path.join(saver_dir, 'model.pth'))

                print("the train file {0}  the train mean loss is {1}".format(file, total_loss / size_input))

            if epoch > 20 and epoch % 5 == 2:
                valid_file = os.listdir(valid_input_dir)[0]
                valid_file = os.path.join(valid_input_dir, valid_file)
                x = np.load(valid_file).astype(np.float32)
                x = np.reshape(x, [x.shape[0], 40, 68, 1])
                size_input = int(x.shape[0] / epoch_size) + 1
                np.random.shuffle(x)
                valid_total_loss = 0
                for i in range(size_input - 1):
                    input_data = torch.tensor(x[i * epoch_size:i * epoch_size + epoch_size, :, 0:40])
                    truth = torch.tensor(x[i * epoch_size:i * epoch_size + epoch_size, 0:28, 40:68])

                    output = res_block(input_data)
                    loss = MSE_loss(output, truth)

                    valid_total_loss += loss.item()
                    print("in the %sth iteration %sth step, the validating loss is %f" % (epoch, i, loss.item()))

                temp_mean_valid_loss = valid_total_loss / size_input
                print(temp_mean_valid_loss)
                if temp_mean_valid_loss > mean_valid_loss:
                    raise Exception("error is small!")
                mean_valid_loss = temp_mean_valid_loss

    except Exception as e:
        print(e)
    finally:
        torch.save(res_block.state_dict(), os.path.join(saver_dir, 'model.pth'))
        print("training is over...")
