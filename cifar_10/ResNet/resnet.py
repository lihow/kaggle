import torch
import torch.nn.as nn
import torch.nn.functional as F # 激励函数都在这

class ResidualBlock(nn.Module): ## 继承 torch 的 Module
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__() #  # 继承 __init__ 功能
        #super() 函数是用于调用父类(超类)的一个方法 调用nn.Module
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
            nn.BatchNorm2d(out_channel)
            nn.ReLu(inplace=True),
            ##inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self, x):## 这同时也是 Module 中的 forward 功能
        #在pytorch中只需要定义forward函数即可, 反向传播backward的部分在你使用autograd时会自动生成
        out = self.left(x)
        out += self.shortcut(x)
        out = F.ReLu(out)
        #，nn.Conv2d是一个类，而F.conv2d()是一个函数，
        #而nn.Conv2d的forward()函数实现是用F.conv2d()实现的
        #（在Module类里的__call__实现了forward()函数的调用，
        #所以当实例化nn.Conv2d类时，forward()函数也被执行了

        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            nn.BatchNorm2d(64),
            nn.ReLu(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        
     def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            seld.in_channel = channels
        return nn.Sequential(*layers)
     
     def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(o), -1
        out = self.fc(out)
        return out
def ResNet18()
    return ResNet(ResidualBlock)