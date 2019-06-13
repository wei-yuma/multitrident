from data import *
from utils.augmentations import SSDAugmentation
from layers.modules.multitrident_multibox_loss import multitridentMultiBoxLoss
from layers.modules.refinedet_multibox_loss import RefineDetMultiBoxLoss
#from ssd import build_ssd
from models.multitrident_refinedet import  build_multitridentrefinedet
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils.logging import Logger
import matplotlib.pyplot as plt
import math
import globalValue
from layers.box_utils import match, log_sum_exp, refine_match_return_matches, scaleAssign

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--input_size', default='320', choices=['320', '512'],
                    type=str, help='RefineDet320 or RefineDet512')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--basenetBN', default='./weights/vgg16_bn-6c64b313.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.2*1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/experiment/320*320/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--withBN', default=False)
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# sys.stdout = Logger(os.path.join(args.save_folder, 'log.txt'))

def train():

    cfg = voc_refinedet["exp"]
    dataset = ExpVOCDetection(root=args.dataset_root,
                           transform=None)

    # im_names = "000069.jpg"
    # image_file = '/home/amax/data/VOCdevkit/VOC2007/JPEGImages/' + im_names
    # image = cv2.imread(image_file, cv2.IMREAD_COLOR)  # uncomment if dataset not download


    refinedet_net = build_multitridentrefinedet('train', cfg['min_dim'], cfg['num_classes'])
    net = refinedet_net
    print(net)
    #input()

    if args.cuda:
        net = torch.nn.DataParallel(refinedet_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        refinedet_net.load_weights(args.resume)
    else:
        if args.withBN:
            vgg_bn_weights = torch.load(args.basenetBN)
            print('Loading base network...')
            model_dict = refinedet_net.vgg.state_dict()
            pretrained_dict = {k: v for k, v in vgg_bn_weights.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            refinedet_net.vgg.load_state_dict(model_dict)
        else:
            # vgg_weights = torch.load(args.save_folder + args.basenet)
            vgg_weights = torch.load(args.basenet)
            print('Loading base network...')
            refinedet_net.vgg.load_state_dict(vgg_weights)


    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        refinedet_net.extras.apply(weights_init)
        refinedet_net.arm_loc.apply(weights_init)
        refinedet_net.arm_conf.apply(weights_init)
        refinedet_net.trm_loc.apply(weights_init)
        refinedet_net.trm_conf.apply(weights_init)
        refinedet_net.branch_for_arm0.apply(bottleneck_init)
        refinedet_net.branch_for_arm1.apply(bottleneck_init)
        refinedet_net.branch_for_arm2.apply(bottleneck_init)
        refinedet_net.branch_for_arm3.apply(bottleneck_init)
        refinedet_net.tcb0.apply(weights_init)
        refinedet_net.tcb1.apply(weights_init)
        refinedet_net.tcb2.apply(weights_init)

        refinedet_net.se0.apply(weights_init)
        refinedet_net.se1.apply(weights_init)
        refinedet_net.se2.apply(weights_init)
        refinedet_net.se3.apply(weights_init)
        # refinedet_net.decov.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    arm_criterion = RefineDetMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    trm_criterion = multitridentMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda, use_ARM=True, use_multiscale=True)

    net.train()
    # loss counters
    arm_loc_loss = 0
    arm_conf_loss = 0
    trm_loc_s_loss = 0
    trm_loc_m_loss = 0
    trm_loc_b_loss = 0
    trm_conf_s_loss = 0
    trm_conf_m_loss = 0
    trm_conf_b_loss = 0
    epoch = 0
    print('Loading the dataset...')

    # epoch_size = len(dataset) // args.batch_size
    # print('Training RefineDet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0


    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    num_all = np.array(0)
    num_small = np.array(0)
    num_middle = np.array(0)
    num_big = np.array(0)
    batch_iterator = iter(data_loader)

    for iteration in range(args.start_iter, cfg['max_iter']):

        # reset epoch loss counters
        arm_loc_loss = 0
        arm_conf_loss = 0
        trm_loc_s_loss = 0
        trm_loc_m_loss = 0
        trm_loc_b_loss = 0
        trm_conf_s_loss = 0
        trm_conf_m_loss = 0
        trm_conf_b_loss = 0
        epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index, iteration)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        # if dataset.getmyimg() != []:
        #     plt.imshow(dataset.getmyimg())
        #     plt.show()
        img = np.array(images)[0].transpose(1,2,0)
        # cv2.imshow("image",img)
        # cv2.waitKey(0)

        images = images.type(torch.FloatTensor)
        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = images
            targets = [ann for ann in targets]




        # forward
        t0 = time.time()
        out = net(images)

        arm_loc_data, arm_conf_data, trm_loc_data1, trm_conf_data1, trm_loc_data2, trm_conf_data2, trm_loc_data3, trm_conf_data3, priors = out
        use_ARM=False
        threshold=0.5
        pos_for_small = torch.ByteTensor(1, 6375)
        pos_for_middle = torch.ByteTensor(1, 6375)
        pos_for_big = torch.ByteTensor(1, 6375)
        loc_t = torch.Tensor(1, 6375, 4)
        conf_t = torch.LongTensor(1, 6375)
        matches_list = torch.Tensor(1, 6375, 4)
        defaults_list = torch.Tensor(1, 6375, 4)
        for idx in range(1):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if True:
                labels = labels >= 0
            defaults = priors.data
            if use_ARM:
                matches, best_pri_overlap, best_pri_idx = refine_match_return_matches(threshold, truths, defaults, cfg['variance'], labels,
                                                      loc_t, conf_t, idx, arm_loc_data[idx].data)
            else:
                matches, best_pri_overlap, best_pri_idx = refine_match_return_matches(threshold, truths, defaults, cfg['variance'], labels,
                                                      loc_t, conf_t, idx)
            matches_list[idx] = matches
            defaults_list[idx] = defaults
            pos_for_small[idx], pos_for_middle[idx], pos_for_big[idx] = scaleAssign(matches, conf_t, idx)  # matc


        # cv2.destroyAllWindows()
        small_gt_set = set(matches_list[pos_for_small])
        middle_gt_set = set(matches_list[pos_for_middle])
        big_gt_set = set(matches_list[pos_for_big])

        small_anchs = defaults_list[pos_for_small]
        middle_anchs = defaults_list[pos_for_middle]
        big_anchs = defaults_list[pos_for_big]

        img_copy = img.copy()
        for rect in small_gt_set:
            cv2.rectangle(img_copy, (rect[0]*320, rect[1]*320), (rect[2]*320, rect[3]*320), (255, 255, 255), 2)
        for rect in middle_gt_set:
            cv2.rectangle(img_copy, (rect[0]*320, rect[1]*320), (rect[2]*320, rect[3]*320), (255, 255, 255), 2)
        for rect in big_gt_set:
            cv2.rectangle(img_copy, (rect[0]*320, rect[1]*320), (rect[2]*320, rect[3]*320), (255, 255, 255), 2)
        for rect in small_anchs:
            x1 = (rect[0]-rect[2]/2)*320
            y1 = (rect[1]-rect[3]/2)*320
            x2 = (rect[0]+rect[2]/2)*320
            y2 = (rect[1]+rect[3]/2)*320
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0,255,0))    # green
        cv2.imshow("image", img_copy)
        cv2.waitKey(1000*2)
        for rect in middle_anchs:
            x1 = (rect[0]-rect[2]/2)*320
            y1 = (rect[1]-rect[3]/2)*320
            x2 = (rect[0]+rect[2]/2)*320
            y2 = (rect[1]+rect[3]/2)*320
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color=(255,0,0))    # blue
        cv2.imshow("image", img_copy)
        cv2.waitKey(1000*2)
        for rect in big_anchs:
            x1 = (rect[0]-rect[2]/2)*320
            y1 = (rect[1]-rect[3]/2)*320
            x2 = (rect[0]+rect[2]/2)*320
            y2 = (rect[1]+rect[3]/2)*320
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color=(0,0,255))  # red
        cv2.imshow("image", img_copy)
        cv2.waitKey(1000*2)

        # backprop
        optimizer.zero_grad()
        arm_loss_l, arm_loss_c = arm_criterion(out, targets)
        trm_loss_s_l, trm_loss_m_l, trm_loss_b_l, trm_loss_s_c, trm_loss_m_c, trm_loss_b_c, n_all, n_small, n_middle, n_big = trm_criterion(out, targets)

        #input()
        arm_loss = arm_loss_l + arm_loss_c
        trm_loss = trm_loss_s_l+ trm_loss_m_l+ trm_loss_b_l+trm_loss_s_c+ trm_loss_m_c+trm_loss_b_c
        loss = arm_loss + trm_loss
        loss.backward()
        # trm_loss.backward()
        optimizer.step()
        t1 = time.time()
        # arm_loc_loss += arm_loss_l.item()
        # arm_conf_loss += arm_loss_c.item()
        # trm_loc_s_loss += trm_loss_s_l.item()
        # trm_loc_m_loss += trm_loss_m_l.item()
        # trm_loc_b_loss += trm_loss_b_l.item()
        # trm_conf_s_loss += trm_loss_s_c.item()
        # trm_conf_m_loss += trm_loss_m_c.item()
        # trm_conf_b_loss += trm_loss_b_c.item()
        num_all = np.append(num_all, n_all)
        num_small = np.append(num_small, n_small)
        num_middle = np.append(num_middle, n_middle)
        num_big = np.append(num_big, n_big)

        if type(trm_loss_s_l) != float:
            trm_loss_s_l_value = trm_loss_s_l.item()
        else:
            trm_loss_s_l_value = trm_loss_s_l
        if type(trm_loss_m_l) != float:
            trm_loss_m_l_value = trm_loss_m_l.item()
        else:
            trm_loss_m_l_value = trm_loss_m_l
        if type(trm_loss_b_l) != float:
            trm_loss_b_l_value = trm_loss_b_l.item()
        else:
            trm_loss_b_l_value = trm_loss_b_l
        if type(trm_loss_s_c) != float:
            trm_loss_s_c_value = trm_loss_s_c.item()
        else:
            trm_loss_s_c_value = trm_loss_s_c
        if type(trm_loss_m_c) != float:
            trm_loss_m_c_value = trm_loss_m_c.item()
        else:
            trm_loss_m_c_value = trm_loss_m_c
        if type(trm_loss_b_c) != float:
            trm_loss_b_c_value = trm_loss_b_c.item()
        else:
            trm_loss_b_c_value = trm_loss_b_c

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || ARM_L: %.4f ARM_C: %.4f TRM_s_L: %.4f TRM_s_C: %.4f TRM_m_L: %.4f TRM_m_C: %.4f TRM_b_L: %.4f TRM_b_C: %.4f ||' \
            % (arm_loss_l.item(), arm_loss_c.item(), trm_loss_s_l_value, trm_loss_s_c_value, trm_loss_m_l_value, trm_loss_m_c_value, trm_loss_b_l_value, trm_loss_b_c_value), end=' ')
            print('\n'+'all:{}  small:{}  middle:{}  big:{} lr:{}'.format(num_all.mean(), num_small.mean(), num_middle.mean(), num_big.mean(), optimizer.param_groups[0]["lr"]))
            num_all = np.array(0)
            num_small = np.array(0)
            num_middle = np.array(0)
            num_big = np.array(0)



        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(refinedet_net.state_dict(), args.save_folder
            + '/RefineDet{}_{}_{}.pth'.format(args.input_size, args.dataset,
            repr(iteration)))
    torch.save(refinedet_net.state_dict(), args.save_folder
            + '/RefineDet{}_{}_final.pth'.format(args.input_size, args.dataset))


def adjust_learning_rate(optimizer, gamma, step, itr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    """multi step lr"""
    lr = args.lr * (gamma ** (step))
    """warmup and cosine lr"""
    # lr = args.lr * math.cos(math.pi/100)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def bottleneck_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        xavier(m.weight.data)
        xavier(m.weight.data)



if __name__ == '__main__':
    train()
