import os
import sys
module_path = os.path.abspath(os.path.join('../models/'))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
import time

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
size = 320
# from refinedet import build_refinedet
# from models.multitrident_refinedet_v2 import  build_multitridentrefinedet
from models.multitrident_refinedet import  build_multitridentrefinedet
net = build_multitridentrefinedet('test', size, 21)    # initialize SSD
# net = build_refinedet('test', 512, 21)
# net.load_weights('../weights/RefineDet512_VOC_final.pth')
# net.load_weights('../weights/experiment/320*320/exp_4_[256relufpn][0.3_0.6][mAP_0.77][dilate:11111-12333-12555]/RefineDet320_VOC_275000.pth')
net.load_weights('../weights/experiment/320*320/RefineDet320_VOC_315000.pth')

"""000210 000111 000144 009539 009589 000069 009539 001275 002333 002338 002341 
002695 002713 003681 003874 003673 003740"""
im_names = "002695.jpg"


image_file = '/home/amax/data/VOCdevkit/VOC2007/JPEGImages/' + im_names
image = cv2.imread(image_file, cv2.IMREAD_COLOR)  # uncomment if dataset not download
#%matplotlib inline
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
img_id = 62
# image = testset.pull_image(img_id)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
plt.figure(figsize=(10,10))
# plt.imshow(rgb_image)
# plt.show()

x = cv2.resize(image, (size, size)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
# plt.imshow(x)
x = torch.from_numpy(x).permute(2, 0, 1)


xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
start = time.time()
y = net(xx)
end = time.time()
print(end-start)

from data import VOC_CLASSES as labels
top_k=100

plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()


detections = y.data


# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    for j in range(detections.size(2)):
        if detections[0,i,j,0] > 0.05:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        else:
            continue
    # j = 0
    # while detections[0,i,j,0] >= -1:
    #     score = detections[0,i,j,0]
    #     label_name = labels[i-1]
    #     display_txt = '%s: %.2f'%(label_name, score)
    #     pt = (detections[0,i,j,1:]*scale).cpu().numpy()
    #     coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
    #     color = colors[i]
    #     currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    #     currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
    #     j+=1

plt.show()




