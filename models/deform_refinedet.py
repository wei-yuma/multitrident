import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from data import voc_refinedet, coco_refinedet
import os
from torch_deform_conv.layers import ConvOffset2D


class RefineDet(nn.Module):
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

    def __init__(self, phase, size, base, extras, ARM, ODM, TCB, num_classes):
        super(RefineDet, self).__init__()
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
        self.arm_deform_loc = nn.ModuleList(ARM[2])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        self.odm_deform_loc = nn.ModuleList(ODM[2])

        #self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])
        self.globalavgpool1 = nn.ModuleList(TCB[3])
        self.globalavgpool2 = nn.ModuleList(TCB[4])
        self.se1 = nn.ModuleList(TCB[5])
        self.se2 = nn.ModuleList(TCB[6])
        # self.channel_att = nn.ModuleList(TCB[7])
        self.sa1 = nn.ModuleList(TCB[8])
        self.sa2 = nn.ModuleList(TCB[9])
        self.spatial_att = nn.ModuleList(TCB[10])
        self.conv = nn.ModuleList(TCB[11])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_RefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500)

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
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()

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

        # apply ARM and ODM to source layers
        for (x, l, dl, c) in zip(sources, self.arm_loc, self.arm_deform_loc, self.arm_conf):
            arm_loc.append(l(dl(x)).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        #print([x.size() for x in sources])
        # calculate TCB features
        #print([x.size() for x in sources])
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(4):
                s = self.tcb0[(3-k)*4 + i](s)
                #print(s.size())
            if k != 0:
                u = p
                u = self.tcb1[3-k](u)
                # print(u.size())
                uu = self.globalavgpool1[3-k](u)
                uu = uu.view(u.size(0), u.size(1))
                for i in range(4):
                    uu = self.se1[(3-k)*4+i](uu)  # from deconv
                uu = uu.view(uu.size(0), uu.size(1), 1, 1)

                ss = self.globalavgpool2[3-k](s)
                ss = ss.view(s.size(0), s.size(1))
                for i in range(4):
                    ss = self.se2[(3-k)*4+i](ss)  # from TL2
                ss = ss.view(ss.size(0), ss.size(1), 1, 1)

                u_se = u*uu
                s_se = s*ss

                u_sa_temp = u_se
                s_sa_temp = s_se
                for i in range(2):
                    u_sa_temp = self.sa1[(3-k)*2+i](u_sa_temp)   # from deconv
                for i in range(2):
                    s_sa_temp = self.sa2[(3-k)*2+i](s_sa_temp)   # from TL2
                cate_sa = torch.cat((u_sa_temp, s_sa_temp), 1)
                cate_sa_temp = cate_sa
                for i in range(2):
                    cate_sa_temp = self.conv[(3-k)*2+i](cate_sa_temp)
                conv_cate_sa = cate_sa_temp
                spatial_att = self.spatial_att[3-k](conv_cate_sa)
                u_spatial_att = spatial_att[:,0,:,:].unsqueeze(1)
                s_spatial_att = spatial_att[:,1,:,:].unsqueeze(1)
                u_sesa = u_se*u_spatial_att
                s_sesa = s_se*s_spatial_att


                s = u_sesa + s_sesa + u

                # s += u
            for i in range(3):
                s = self.tcb2[(3-k)*3 + i](s)
            p = s
            tcb_source.append(s)
        #print([x.size() for x in tcb_source])
        tcb_source.reverse()

        # apply ODM to source layers
        for (x, l, dl, c) in zip(tcb_source, self.odm_loc, self.odm_deform_loc, self.odm_conf):

            odm_loc.append(l(dl(x)).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())

        if self.phase == "test":
            #print(loc, conf)
            output = self.detect(
                arm_loc.view(arm_loc.size(0), -1, 4),           # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,
                             2)),                               # arm conf preds
                odm_loc.view(odm_loc.size(0), -1, 4),           # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size(0), -1,
                             self.num_classes)),                # odm conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
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
    arm_deform_loc_layers = []
    arm_deform_loc_layers = [ConvOffset2D(512), ConvOffset2D(512), ConvOffset2D(1024)]
    vgg_source = [21, 28, -2]
    for k, v in enumerate(vgg_source):

        arm_loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * 2, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        arm_deform_loc_layers += [ConvOffset2D(512)]
        arm_loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers, arm_deform_loc_layers)

# def odm_multibox(vgg, extra_layers, cfg, num_classes):
#     odm_loc_layers = []
#     odm_conf_layers = []
#     vgg_source = [21, 28, -2]
#     for k, v in enumerate(vgg_source):
#         odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
#         odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
#     for k, v in enumerate(extra_layers[1::2], 3):
#         odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
#         odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
#     return (odm_loc_layers, odm_conf_layers)

def odm_multibox(vgg, extra_layers, cfg, num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    odm_deform_loc_layers = []
    # odm_deform_conf_layers = []
    vgg_source = [21, 28, -2]
    for k, v in enumerate(vgg_source):
        odm_deform_loc_layers += [ConvOffset2D(256)]
        # odm_deform_conf_layers += [ConvOffset2D(256)]
        odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        odm_deform_loc_layers += [ConvOffset2D(256)]
        # odm_deform_conf_layers += [ConvOffset2D(256)]
        odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers, odm_deform_loc_layers)

def add_tcb(cfg):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    se_layers_1 = []
    se_layers_2 = []
    avgpool_layers_1 = []
    avgpool_layers_2 = []
    channel_mix_att = []
    spatial_mix_att = []
    sa_layers_1 = []
    sa_layers_2 = []
    conv = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1),
                                 nn.ReLU()
        ]
        feature_pred_layers += [nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True)
        ]
        if k != len(cfg) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]


            avgpool_layers_1 += [nn.AdaptiveAvgPool2d(1)]
            avgpool_layers_2 += [nn.AdaptiveAvgPool2d(1)]
            se_layers_1 += [nn.Linear(in_features=256, out_features=round(256 / 16)),
                          nn.ReLU(inplace=True),
                          nn.Linear(in_features=round(256 / 16), out_features=256),
                            nn.Sigmoid()]
            se_layers_2 += [nn.Linear(in_features=256, out_features=round(256 / 16)),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_features=round(256 / 16), out_features=256),
                            nn.Sigmoid()]
            # channel_mix_att += [nn.Linear(in_features=256*2, out_features=256),
            #                     nn.Linear(in_features=256, out_features=256),
            #                     nn.Sigmoid()]
            sa_layers_1 += [nn.Conv2d(256, 1, 1, padding=0),
                                nn.Sigmoid()]
            sa_layers_2 += [nn.Conv2d(256, 1, 1, padding=0),
                                nn.Sigmoid()]
            spatial_mix_att += [nn.Softmax(dim=1)]
            conv += [nn.Conv2d(2, 2, 1, 1, padding=0),
                     nn.ReLU(inplace=True)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers, avgpool_layers_1, avgpool_layers_2, se_layers_1, se_layers_2, channel_mix_att, sa_layers_1, sa_layers_2, spatial_mix_att, conv)

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


def build_refinedet(phase, size=320, num_classes=21):
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
    ODM_ = odm_multibox(base_, extras_, mbox[str(size)], num_classes)
    TCB_ = add_tcb(tcb[str(size)])
    return RefineDet(phase, size, base_, extras_, ARM_, ODM_, TCB_, num_classes)
