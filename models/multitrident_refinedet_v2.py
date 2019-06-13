import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from layers.functions.detection_multitrident import *
from data import voc_refinedet, coco_refinedet
import os
# from bottleneck.bottleneck import Bottleneck
from bottleneck.shareWeightTrident import trident
class multitridentRefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, ARM, TRM, num_classes, branch, fpn):
        super(multitridentRefineDet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(512, 8)
        self.extras = nn.ModuleList(extras)

        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.trm_loc = nn.ModuleList(TRM[0])
        self.trm_conf = nn.ModuleList(TRM[1])
        #self.tcb = nn.ModuleList(TCB)
        self.fpn0 = nn.ModuleList(fpn[0])
        self.fpn1 = nn.ModuleList(fpn[1])
        self.fpn2 = nn.ModuleList(fpn[2])
        self.fpn3 = nn.ModuleList(fpn[3])
        self.decov = nn.ModuleList(fpn[4])

        self.branch_for_arm0 = nn.ModuleList(branch[0])
        self.branch_for_arm1 = nn.ModuleList(branch[1])
        self.branch_for_arm2 = nn.ModuleList(branch[2])
        self.branch_for_arm3 = nn.ModuleList(branch[3])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_multitridentRefineDet(num_classes, self.size, 0, 200, 0.01, 0.45, 0.01, 500)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        trm_source = list()
        arm_loc = list()
        arm_conf = list()
        trm_loc = list()
        trm_conf = list()

        arm_branch = list()
        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(30):
            x = self.vgg[k](x)
            if 22 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
            elif 29 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)


        sources.reverse()


        a = sources[0].clone()
        for i in range(2):
            a = self.fpn3[i](a)
        fpn_for_arm3 = self.relu(a)             # fpn for arm3
        a = self.decov[0](a)
        b = sources[1].clone()
        for i in range(2):
            b = self.fpn2[i](b)
        fpn_for_arm2 = self.relu(b + a)          #fpn for arm2

        a = self.decov[1](fpn_for_arm2)
        b = sources[2].clone()
        for i in range(2):
            b = self.fpn1[i](b)
        fpn_for_arm1 = self.relu(b + a)          #fpn for arm1

        a = self.decov[2](fpn_for_arm1)
        b = sources[3].clone()
        for i in range(2):
            b = self.fpn0[i](b)
        fpn_for_arm0 = self.relu(b + a)          #fpn for arm0


        trm_source.append(fpn_for_arm0)
        trm_source.append(fpn_for_arm1)
        trm_source.append(fpn_for_arm2)
        trm_source.append(fpn_for_arm3)

        # apply ARM and ODM to source layers
        for (x, l, c) in zip(trm_source, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)

        sources = trm_source
        arm0_b1 = sources[0].clone()
        arm0_b2 = sources[0].clone()
        arm0_b3 = sources[0].clone()
        for i,b in enumerate(self.branch_for_arm0):
            arm0_b1, arm0_b2, arm0_b3 = b(arm0_b1, arm0_b2, arm0_b3)

        arm1_b1 = sources[1].clone()
        arm1_b2 = sources[1].clone()
        arm1_b3 = sources[1].clone()
        for i,b in enumerate(self.branch_for_arm1):
            arm1_b1, arm1_b2, arm1_b3 = b(arm1_b1, arm1_b2, arm1_b3)

        arm2_b1 = sources[2].clone()
        arm2_b2 = sources[2].clone()
        arm2_b3 = sources[2].clone()
        for i,b in enumerate(self.branch_for_arm2):
            arm2_b1, arm2_b2, arm2_b3 = b(arm2_b1, arm2_b2, arm2_b3)

        arm3_b1 = sources[3].clone()
        arm3_b2 = sources[3].clone()
        arm3_b3 = sources[3].clone()
        for i,b in enumerate(self.branch_for_arm3):
            arm3_b1, arm3_b2, arm3_b3 = b(arm3_b1, arm3_b2, arm3_b3)

        arm_branch.append(arm0_b1)
        arm_branch.append(arm1_b1)
        arm_branch.append(arm2_b1)
        arm_branch.append(arm3_b1)
        arm_branch.append(arm0_b2)
        arm_branch.append(arm1_b2)
        arm_branch.append(arm2_b2)
        arm_branch.append(arm3_b2)
        arm_branch.append(arm0_b3)
        arm_branch.append(arm1_b3)
        arm_branch.append(arm2_b3)
        arm_branch.append(arm3_b3)

        for x,l,c in zip(arm_branch, self.trm_loc, self.trm_conf):
            trm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            trm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        trm_loc1 = torch.cat([o.view(o.size(0), -1) for o in trm_loc[0:4]], 1)
        trm_conf1 = torch.cat([o.view(o.size(0), -1) for o in trm_conf[0:4]], 1)
        trm_loc2 = torch.cat([o.view(o.size(0), -1) for o in trm_loc[4:8]], 1)
        trm_conf2 = torch.cat([o.view(o.size(0), -1) for o in trm_conf[4:8]], 1)
        trm_loc3 = torch.cat([o.view(o.size(0), -1) for o in trm_loc[8:12]], 1)
        trm_conf3 = torch.cat([o.view(o.size(0), -1) for o in trm_conf[8:12]], 1)


        # #print([x.size() for x in sources])
        # # calculate TCB features
        # #print([x.size() for x in sources])
        # p = None
        # for k, v in enumerate(sources[::-1]):
        #     s = v
        #     for i in range(3):
        #         s = self.tcb0[(3-k)*3 + i](s)
        #         #print(s.size())
        #     if k != 0:
        #         u = p
        #         u = self.tcb1[3-k](u)
        #         s += u
        #     for i in range(3):
        #         s = self.tcb2[(3-k)*3 + i](s)
        #     p = s
        #     tcb_source.append(s)
        # #print([x.size() for x in tcb_source])
        # tcb_source.reverse()
        #
        # # apply ODM to source layers
        # for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
        #     odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        #     odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        # odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        # #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())

        if self.phase == "test":
            #print(loc, conf)
            output = self.detect(
                arm_loc.view(arm_loc.size(0), -1, 4),           # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,
                             2)),                               # arm conf preds
                trm_loc1.view(trm_loc1.size(0), -1, 4),           # trm loc preds
                self.softmax(trm_conf1.view(trm_conf1.size(0), -1,
                             self.num_classes)),                # trm conf preds
                trm_loc2.view(trm_loc2.size(0), -1, 4),  # trm loc preds
                self.softmax(trm_conf2.view(trm_conf2.size(0), -1,
                                            self.num_classes)),  # trm conf preds
                trm_loc3.view(trm_loc3.size(0), -1, 4),  # trm loc preds
                self.softmax(trm_conf3.view(trm_conf3.size(0), -1,
                                            self.num_classes)),  # trm conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                trm_loc1.view(trm_loc1.size(0), -1, 4),
                trm_conf1.view(trm_conf1.size(0), -1, self.num_classes),
                trm_loc2.view(trm_loc2.size(0), -1, 4),
                trm_conf2.view(trm_conf2.size(0), -1, self.num_classes),
                trm_loc3.view(trm_loc3.size(0), -1, 4),
                trm_conf3.view(trm_conf3.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def arm_multibox(vgg, extra_layers, cfg):
    arm_loc_layers = []
    arm_conf_layers = []
    vgg_source = [21, 28, -2]
    for k, v in enumerate(vgg_source):
        arm_loc_layers += [nn.Conv2d(256,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(256,
                        cfg[k] * 2, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        arm_loc_layers += [nn.Conv2d(256, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(256, cfg[k]
                                  * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)

def trident_multibox(num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    for i in range(12):
        odm_loc_layers += [nn.Conv2d(256, 3 * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)

def add_tcb(cfg):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1)
        ]
        feature_pred_layers += [nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True)
        ]
        if k != len(cfg) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)

base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '320': [256, 'S', 512],
    '512': [256, 'S', 512],
}
mbox = {
    '320': [3, 3, 3, 3],  # number of boxes per feature map location
    '512': [3, 3, 3, 3],  # number of boxes per feature map location
}

tcb = {
    '320': [512, 512, 1024, 512],
    '512': [512, 512, 1024, 512],
}

def fpn():
    for_arm3 = [nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1),
                ]
    for_arm2 = [nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1),
                ]
    for_arm1 = [nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1),
                ]
    for_arm0 = [nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1)]
    deconves = [nn.ConvTranspose2d(256, 256, 2, 2), nn.ConvTranspose2d(256, 256, 2, 2), nn.ConvTranspose2d(256, 256, 2, 2)]
    return (for_arm0, for_arm1, for_arm2, for_arm3, deconves)



def tridentbranch():

    branch_for_arm0 = [
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 1], dilate=[1, 1, 1]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 2, 2], dilate=[1, 2, 2]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 5], dilate=[1, 1, 5]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 2, 1], dilate=[1, 2, 1]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 2], dilate=[1, 1, 2]),
    ]
    branch_for_arm1 = [
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 1], dilate=[1, 1, 1]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 2, 2], dilate=[1, 2, 2]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 5], dilate=[1, 1, 5]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 2, 1], dilate=[1, 2, 1]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 2], dilate=[1, 1, 2]),
                         ]
    branch_for_arm2 = [
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 1], dilate=[1, 1, 1]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 2, 2], dilate=[1, 2, 2]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 5], dilate=[1, 1, 5]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 2, 1], dilate=[1, 2, 1]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 2], dilate=[1, 1, 2]),
                      ]
    branch_for_arm3 = [
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 1], dilate=[1, 1, 1]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 2, 2], dilate=[1, 2, 2]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 5], dilate=[1, 1, 5]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 2, 1], dilate=[1, 2, 1]),
        trident(inplanes=256, planes=64, stride=1, downsample=None, padding=[1, 1, 2], dilate=[1, 1, 2]),
                      ]
    return (branch_for_arm0, branch_for_arm1, branch_for_arm2, branch_for_arm3)



def build_multitridentrefinedet(phase, size=320, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only RefineDet320 and RefineDet512 is supported!")
        return
    base_ = vgg(base[str(size)], 3)
    extras_ = add_extras(extras[str(size)], size, 1024)
    ARM_ = arm_multibox(base_, extras_, mbox[str(size)])
    TRM_ = trident_multibox(num_classes)
    FPN_ = fpn()
    branch_ = tridentbranch()
    return multitridentRefineDet(phase, size, base_, extras_, ARM_, TRM_, num_classes, branch_, FPN_)
