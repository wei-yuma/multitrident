# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from data import coco as cfg
# print(os.getcwd())
from layers.box_utils import match, log_sum_exp, refine_match_return_matches, scaleAssign
import globalValue

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class multitridentMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, theta=0.01, use_ARM=False, use_multiscale=True):
        super(multitridentMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.theta = theta
        self.use_ARM = use_ARM
        self.use_multiscale = use_multiscale


    def computeSmothL1Loss(self, pos_for_WHAT, loc_pred, loc_thruth):
        num_pos = pos_for_WHAT.long().sum(1, keepdim=True)
        if num_pos.sum() == 0:
            loss_l_for_WHAT = 0
            return loss_l_for_WHAT
        pos_for_WHAT_idx = pos_for_WHAT.unsqueeze(pos_for_WHAT.dim()).expand_as(loc_pred)
        loc_p1 = loc_pred[pos_for_WHAT_idx].view(-1, 4)
        loc_t = loc_thruth[pos_for_WHAT_idx].view(-1, 4)
        loss_l_for_WHAT = F.smooth_l1_loss(loc_p1, loc_t, reduction='sum')
        return loss_l_for_WHAT

    def computeCrossEntropy(self, loss_c_for_WHAT, num_batch, pos_for_WHAT, conf_data, conf_truth):
        num_pos = pos_for_WHAT.long().sum(1, keepdim=True)
        if num_pos.sum() == 0:
            c_loss_for_WHAT = 0
            return c_loss_for_WHAT, num_pos

        loss_c_for_WHAT[pos_for_WHAT.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c_for_small = loss_c_for_WHAT.view(num_batch, -1)
        _, loss_idx = loss_c_for_small.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # num_pos = pos_for_WHAT.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos_for_WHAT.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # neglong = neg.long()
        # print("==========================================")
        # print("pos-->", num_pos.data.sum(), "neg-->",num_neg.data.sum(), "  ",neg.data.sum())

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos_for_WHAT.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        """do hard negmining"""
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_truth[(pos_for_WHAT + neg).gt(0)]
        """do not do hard negmining"""
        # conf_p = conf_data[(pos_idx).gt(0)].view(-1, self.num_classes)
        # targets_weighted = conf_truth[(pos_for_WHAT).gt(0)]

        c_loss_for_WHAT = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        num_pos_for_WHAT = num_pos
        return c_loss_for_WHAT, num_pos_for_WHAT

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        arm_loc_data, arm_conf_data, trm_loc_data1, trm_conf_data1, trm_loc_data2, trm_conf_data2, trm_loc_data3, trm_conf_data3, priors = predictions
        #print(arm_loc_data.size(), arm_conf_data.size(),
        #      odm_loc_data.size(), odm_conf_data.size(), priors.size())
        #input()
        if self.use_ARM:
            loc_data1, conf_data1 = trm_loc_data1, trm_conf_data1
            loc_data2, conf_data2 = trm_loc_data2, trm_conf_data2
            loc_data3, conf_data3 = trm_loc_data3, trm_conf_data3
        # assert loc_data1.size == loc_data2.size == loc_data3.size
        num = loc_data1.size(0)
        priors = priors[:loc_data1.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        #print(loc_data.size(), conf_data.size(), priors.size())
        # init valid_scale_index
        pos_for_small = torch.ByteTensor(num, num_priors)
        pos_for_middle = torch.ByteTensor(num, num_priors)
        pos_for_big = torch.ByteTensor(num, num_priors)
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        matches_list = torch.Tensor(num, num_priors, 4)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if num_classes == 2:
                labels = labels >= 0
            defaults = priors.data
            if self.use_ARM:
                matches, _, _ = refine_match_return_matches(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx, arm_loc_data[idx].data)
            else:
                matches, _, _ = refine_match_return_matches(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)
            matches_list[idx] = matches
            pos_for_small[idx], pos_for_middle[idx], pos_for_big[idx] = scaleAssign(matches, conf_t, idx)  # matches: using ARM loc as priors to match with pred loc
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        #loc_t = Variable(loc_t, requires_grad=False)
        #conf_t = Variable(conf_t, requires_grad=False)
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        #print(loc_t.size(), conf_t.size())


        if self.use_ARM:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_tmp = P[:,:,1]
            object_score_index = arm_conf_tmp <= self.theta
            pos_for_small[object_score_index.data] = 0

            pos_for_middle[object_score_index.data] = 0

            pos_for_big[object_score_index.data] = 0
            pos = conf_t > 0
            pos[object_score_index.data] = 0
            if not self.use_multiscale:
                pos_for_small = pos
                pos_for_middle = pos
                pos_for_big = pos

        pos_for_small = pos_for_small.cuda()
        pos_for_middle = pos_for_middle.cuda()
        pos_for_big = pos_for_big.cuda()

        #print(pos.size())
        #num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        """look small anchors are where
        """
        # globalValue.addItem("small_gt_set", set(matches_list[pos_for_small]))
        # globalValue.addItem("mid_gt_set", set(matches_list[pos_for_middle]))
        # globalValue.addItem("big_gt_set", set(matches_list[pos_for_big]))


        loss_l_for_small = self.computeSmothL1Loss(pos_for_WHAT=pos_for_small, loc_pred=loc_data1, loc_thruth=loc_t)
        loss_l_for_middle = self.computeSmothL1Loss(pos_for_WHAT=pos_for_middle, loc_pred=loc_data2, loc_thruth=loc_t)
        loss_l_for_big = self.computeSmothL1Loss(pos_for_WHAT=pos_for_big, loc_pred=loc_data3, loc_thruth=loc_t)
        '''
        pos_for_middle_idx = pos_for_middle.unsqueeze(pos_for_middle.dim()).expand_as(loc_data2)
        loc_p2 = loc_data2[pos_for_middle_idx].view(-1, 4)
        loc_t = loc_t[pos_for_middle_idx].view(-1, 4)
        loss_l_for_middle = F.smooth_l1_loss(loc_p2, loc_t, reduction='sum')

        pos_for_big_idx = pos_for_big.unsqueeze(pos_for_big.dim()).expand_as(loc_data3)
        loc_p3 = loc_data1[pos_for_big_idx].view(-1, 4)
        loc_t = loc_t[pos_for_big_idx].view(-1, 4)
        loss_l_for_big = F.smooth_l1_loss(loc_p3, loc_t, reduction='sum')
        '''

        # Compute max conf across batch for hard negative mining
        conf_t_for_small = conf_t.clone()
        conf_t_for_small[pos_for_small.eq(0)] = 0
        conf_t_for_middle = conf_t.clone()
        conf_t_for_middle[pos_for_middle.eq(0)] = 0
        conf_t_for_big = conf_t.clone()
        conf_t_for_big[pos_for_big.eq(0)] = 0

        batch_conf = conf_data1.view(-1, self.num_classes)
        loss_c_for_small = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t_for_small.view(-1, 1))
        batch_conf = conf_data2.view(-1, self.num_classes)
        loss_c_for_middle = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t_for_middle.view(-1, 1))
        batch_conf = conf_data3.view(-1, self.num_classes)
        loss_c_for_big = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t_for_big.view(-1, 1))
        #print(loss_c.size())



        loss_conf_for_small, num_pos_for_small = self.computeCrossEntropy(loss_c_for_WHAT=loss_c_for_small, num_batch=num,
                                                       pos_for_WHAT=pos_for_small, conf_data=conf_data1,
                                                       conf_truth=conf_t_for_small)
        loss_conf_for_middle, num_pos_for_middle = self.computeCrossEntropy(loss_c_for_WHAT=loss_c_for_middle, num_batch=num,
                                                        pos_for_WHAT=pos_for_middle, conf_data=conf_data2,
                                                        conf_truth=conf_t_for_middle)
        loss_conf_for_big, num_pos_for_big = self.computeCrossEntropy(loss_c_for_WHAT=loss_c_for_big, num_batch=num,
                                                        pos_for_WHAT=pos_for_big, conf_data=conf_data3,
                                                        conf_truth=conf_t_for_big)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        num_pos = pos.long().sum(1,keepdim=True)
        N_for_all = num_pos.data.sum().float()
        N_for_small = num_pos_for_small.data.sum().float()
        N_for_middle = num_pos_for_middle.data.sum().float()
        N_for_big = num_pos_for_big.data.sum().float()

        N_for_small = max(N_for_small, 0.0001)
        N_for_middle = max(N_for_middle, 0.0001)
        N_for_big = max(N_for_big, 0.0001)

        # print('all:{}  small:{}  middle:{}  big:{}'.format(N_for_all,N_for_small,N_for_middle,N_for_big))
        # N = N_for_small+N_for_middle+N_for_big
        #N = max(num_pos.data.sum().float(), 1)

        # loss_l_for_small /= N_for_small
        # loss_l_for_middle /= N_for_middle
        # loss_l_for_big /= N_for_big

        # loss_l = loss_l_for_small + loss_l_for_middle + loss_l_for_big

        # loss_conf_for_small /= N_for_small
        # loss_conf_for_middle /= N_for_middle
        # loss_conf_for_big /= N_for_big

        # loss_c = loss_conf_for_small + loss_conf_for_middle + loss_conf_for_big

        #print(N, loss_l, loss_c)
        return loss_l_for_small/N_for_small, \
               loss_l_for_middle/N_for_middle, \
               loss_l_for_big/N_for_big, \
               loss_conf_for_small/N_for_small, \
               loss_conf_for_middle/N_for_middle,\
               loss_conf_for_big/N_for_big,  N_for_all, N_for_small, N_for_middle, N_for_big


# # predictions
# arm_loc = torch.rand((4,100,4))
# arm_conf = torch.rand((4,100,2))
# odm_loc1 = torch.rand((4,100,4))
# odm_loc2 = torch.rand((4,100,4))
# odm_loc3 = torch.rand((4,100,4))
# odm_conf1 = torch.rand((4,100,21))
# odm_conf2 = torch.rand((4,100,21))
# odm_conf3 = torch.rand((4,100,21))
# anchor = torch.rand((100,4))
# # ground truths
# gt1 = torch.Tensor([[0.56,0.42,0.8,0.5,14.]])
# gt2 = torch.Tensor([[0.23,0.24,0.56,0.34,7.]])
# gt3 = torch.Tensor([[0.4527,0.0516,0.4938,0.1463,15.],
#                     [0.3247,0.0516,0.7708,0.5237,14.0]])
# gt4 = torch.Tensor([[0.4863,0.3579,0.7280,0.8428,11.0]])
# # put them together
# truths = [gt1,gt2,gt3,gt4]
# preds = (arm_loc, arm_conf, odm_loc1, odm_conf1, odm_loc2, odm_conf2, odm_loc3, odm_conf3, anchor)
# # init a lossfunction
# lossfunction = multitridentMultiBoxLoss(21, 0.0, True, 0, True, 3, 0.5,
#                              False, False, use_ARM=True)
# loss = lossfunction.forward(preds, truths)
# print(loss)