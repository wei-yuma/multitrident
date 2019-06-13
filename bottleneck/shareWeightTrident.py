import torch.nn as nn
import torch

class trident(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, padding=[0, 0, 0], dilate=[1, 1, 1]):
        super(trident, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilate = dilate
        self.downsample = downsample
        self.share_weight4conv1 = nn.Parameter(torch.randn(planes, inplanes, 1, 1))
        self.share_weight4conv2 = nn.Parameter(torch.randn(planes, planes, 3, 3))
        self.share_weight4conv3 = nn.Parameter(torch.randn(planes * self.expansion, planes, 1, 1))

        self.bn11 = nn.BatchNorm2d(planes)
        self.bn12 = nn.BatchNorm2d(planes)
        self.bn13 = nn.BatchNorm2d(planes * self.expansion)

        self.bn21 = nn.BatchNorm2d(planes)
        self.bn22 = nn.BatchNorm2d(planes)
        self.bn23 = nn.BatchNorm2d(planes * self.expansion)

        self.bn31 = nn.BatchNorm2d(planes)
        self.bn32 = nn.BatchNorm2d(planes)
        self.bn33 = nn.BatchNorm2d(planes * self.expansion)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)


    def forward_once_for_small(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn11(out)
        out = self.relu1(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, stride=self.stride, padding=self.padding[0], bias=None, dilation=self.dilate[0])

        out = self.bn12(out)
        out = self.relu1(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)
        out = self.bn13(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu1(out)

        return out

    def forward_once_for_middle(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn21(out)
        out = self.relu2(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, stride=self.stride, padding=self.padding[1], bias=None,
                                   dilation=self.dilate[1])

        out = self.bn22(out)
        out = self.relu2(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)
        out = self.bn23(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out

    def forward_once_for_big(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn31(out)
        out = self.relu3(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, stride=self.stride, padding=self.padding[2], bias=None,
                                   dilation=self.dilate[2])

        out = self.bn32(out)
        out = self.relu3(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)
        out = self.bn33(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out

    def forward(self, input_for_small, input_for_middle, input_for_big):
        for_small = self.forward_once_for_small(input_for_small)
        for_middle = self.forward_once_for_middle(input_for_middle)
        for_big = self.forward_once_for_big(input_for_big)
        return for_small, for_middle, for_big