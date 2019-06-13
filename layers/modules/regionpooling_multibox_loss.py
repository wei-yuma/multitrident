# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp, refine_match
# from RoIAlign_pytorch.roi_align.roi_align import RoIAlign
from RoIAlign2_pytorch.roialign.roi_align import ROIAlign
# from RoIAlign.roi_align.roi_align import CropAndResize
from ..box_utils import decode, nms, center_size

class RPRefineDetMultiBoxLoss(nn.Module):
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
                 use_gpu=True, theta=0.01, use_ARM=False):
        super(RPRefineDetMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.theta = theta
        self.use_ARM = use_ARM

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
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors, priorsdata, tcbsource = predictions
        # arm_loc_data, arm_conf_data, _, _, _, _, _, _, priors = predictions
        #print(arm_loc_data.size(), arm_conf_data.size(),
        #      odm_loc_data.size(), odm_conf_data.size(), priors.size())
        #input()

        # for (x, rp) in zip(tcb_source, self.region_pool):
        #     r_p.append(roi_align(x, ))
        if self.use_ARM:
            loc_data, conf_data = odm_loc_data, odm_conf_data

        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        #print(loc_data.size(), conf_data.size(), priors.size())

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if num_classes == 2:
                labels = labels >= 0
            defaults = priors.data
            if self.use_ARM:
                refine_match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx, arm_loc_data[idx].data)
            else:
                refine_match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)
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
            pos = conf_t > 0
            pos[object_score_index.data] = 0
        else:
            pos = conf_t > 0
        #print(pos.size())
        #num_pos = pos.sum(dim=1, keepdim=True)

        if self.use_ARM:
            num = loc_data.size(0)
            # h = tcbsource[0].size(2)
            r_p_feature = torch.Tensor(num, num_priors, 256)
            boxes = torch.Tensor(num, num_priors, 5)
            scale = [1/8, 1/16, 1/32, 1/64]
            input_size = 320
            num_boxes1 = 40 * 40 * 3
            num_boxes2 = 20 * 20 * 3
            num_boxes3 = 10 * 10 * 3
            num_boxes4 = 5 * 5 * 3
            for i in range(num):
                default = decode(arm_loc_data[i, :, :], priorsdata, self.variance)
                default = center_size(default)
                decoded_boxes = decode(loc_data[i, :, :], default, self.variance)
                dboxes = decoded_boxes
                dboxes = dboxes * input_size
                batch_idx = torch.Tensor([i]).float().expand_as(dboxes)
                batch_idx = batch_idx[:, 0].unsqueeze(1)
                dboxes = torch.cat([batch_idx, dboxes], 1).contiguous()
                dboxes = dboxes.cuda()
                boxes[i] = dboxes
            feat_0 = tcbsource[0].clone()
            feat_1 = tcbsource[1].clone()
            feat_2 = tcbsource[2].clone()
            feat_3 = tcbsource[3].clone()
            feat_0 = feat_0.detach().cuda()
            feat_0.requires_grad = True
            feat_1 = feat_1.detach().cuda()
            feat_1.requires_grad = True
            feat_2 = feat_2.detach().cuda()
            feat_2.requires_grad = True
            feat_3 = feat_3.detach().cuda()
            feat_3.requires_grad = True
            roi_align_scale_list=[]
            roi_align_batch_list = []
            for i in range(num):
                for s in scale:
                    roi_align_scale_list.append(ROIAlign(1, 1, s, 0))
                roi_align_batch_list.append(roi_align_scale_list)
            for i in range(num):


                for k,s in enumerate(scale):
                    roi_align = roi_align_batch_list[i][k]
                    if s == 1 / 8:
                        dboxes_0 = boxes[i][ : num_boxes1, :]
                        temp_0 = roi_align(feat_0, dboxes_0)
                    if s == 1 / 16:
                        dboxes_1 = boxes[i][num_boxes1 : num_boxes1 + num_boxes2 , :]
                        temp_1 = roi_align(feat_1, dboxes_1)
                    if s == 1 / 32:
                        dboxes_2 = boxes[i][num_boxes1 + num_boxes2 : num_boxes1 + num_boxes2 + num_boxes3, :]
                        temp_2 = roi_align(feat_2, dboxes_2)
                    if s == 1 / 64:
                        dboxes_3 = boxes[i][num_boxes1 + num_boxes2 + num_boxes3 : , :]
                        temp_3 = roi_align(feat_3, dboxes_3)
                        pass
                temp = torch.cat([temp_0,temp_1,temp_2,temp_3],0)
                r_p_feature[i] = temp.squeeze(3).squeeze(2)




        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        #print(loss_c.size())

        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        #print(num_pos.size(), num_neg.size(), neg.size())

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        #print(pos_idx.size(), neg_idx.size(), conf_p.size(), targets_weighted.size())
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        #N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        #print(N, loss_l, loss_c)
        return loss_l, loss_c
