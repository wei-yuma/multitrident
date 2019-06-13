import torch.nn as nn
import torch

class branch_for_small(nn.Module):

    def __init__(self, inplanes, planes):
        super(branch_for_small, self).__init__()
        self.share_weight4conv1 = nn.Parameter(torch.randn(planes, inplanes, 1, 1))
        self.share_weight4conv2 = nn.Parameter(torch.randn(planes, planes, 3, 3))
        self.share_weight4conv3 = nn.Parameter(torch.randn(planes, planes, 1, 1))
        self.share_weight4conv4 = nn.Parameter(torch.randn(planes, planes, 3, 3))
        self.share_weight4conv5 = nn.Parameter(torch.randn(planes, planes, 1, 1))
        self.share_weight4conv6 = nn.Parameter(torch.randn(planes, planes, 3, 3))
        self.share_weight4conv7 = nn.Parameter(torch.randn(planes, planes, 1, 1))
        self.share_weight4conv8 = nn.Parameter(torch.randn(planes, planes, 3, 3))

        self.bias1 = nn.Parameter(torch.zeros(planes))
        self.bias2 = nn.Parameter(torch.zeros(planes))
        self.bias3 = nn.Parameter(torch.zeros(planes))
        self.bias4 = nn.Parameter(torch.zeros(planes))
        self.bias5 = nn.Parameter(torch.zeros(planes))
        self.bias6 = nn.Parameter(torch.zeros(planes))
        self.bias7 = nn.Parameter(torch.zeros(planes))
        self.bias8 = nn.Parameter(torch.zeros(planes))


        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(planes)
        self.bn5 = nn.BatchNorm2d(planes)
        self.bn6 = nn.BatchNorm2d(planes)
        self.bn7 = nn.BatchNorm2d(planes)
        self.bn8 = nn.BatchNorm2d(planes)



        self.relu = nn.ReLU(inplace=True)



    def forward_once_for_small(self, x):

        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=self.bias1)
        out = self.bn1(out)
        out = self.relu(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, stride=1, padding=1, bias=self.bias2, dilation=1)
        out = self.bn2(out)
        out = self.relu(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=self.bias3)
        out = self.bn3(out)
        out = self.relu(out)

        out = nn.functional.conv2d(out, self.share_weight4conv4, stride=1, padding=1, bias=self.bias4, dilation=1)
        out = self.bn4(out)
        out = self.relu(out)

        out = nn.functional.conv2d(out, self.share_weight4conv5, bias=self.bias5)
        out = self.bn5(out)
        out = self.relu(out)

        out = nn.functional.conv2d(out, self.share_weight4conv6, stride=1, padding=1, bias=self.bias6, dilation=1)
        out = self.bn6(out)
        out = self.relu(out)

        out = nn.functional.conv2d(out, self.share_weight4conv7, bias=self.bias7)
        out = self.bn7(out)
        out = self.relu(out)

        out = nn.functional.conv2d(out, self.share_weight4conv8, stride=1, padding=1, bias=self.bias8, dilation=1)
        out = self.bn8(out)
        out = self.relu(out)

        return out


    def forward(self, input_for_small):
        for_small = self.forward_once_for_small(input_for_small)
        return for_small