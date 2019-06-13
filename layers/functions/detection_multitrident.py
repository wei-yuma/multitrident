import torch
from torch.autograd import Function
from ..box_utils import decode, nms, center_size
from data import voc_refinedet as cfg


class Detect_multitridentRefineDet(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh,
                 objectness_thre, keep_top_k):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.objectness_thre = objectness_thre
        self.variance = cfg[str(size)]['variance']

    """way 2"""
    def chooseMaxScore(self,loc_data1, conf_data1, loc_data2, conf_data2, loc_data3, conf_data3, prior, nobj_idx):
        conf_data1[nobj_idx.expand_as(conf_data1)] = 0
        conf_data2[nobj_idx.expand_as(conf_data2)] = 0
        conf_data3[nobj_idx.expand_as(conf_data3)] = 0

        num = loc_data1.size(0)  # batch size
        num_priors = prior.size(0)
        choose_result_loc = torch.zeros(num, num_priors, 4)
        choose_result_conf = torch.zeros(num, num_priors, 21)

        for i in range(num):
            loc1 = loc_data1[i].clone().unsqueeze(0)
            loc2 = loc_data2[i].clone().unsqueeze(0)
            loc3 = loc_data3[i].clone().unsqueeze(0)
            loc_temp = torch.cat([loc1, loc2, loc3],0)
            conf_scores1 = conf_data1[i].clone().unsqueeze(0)
            conf_scores2 = conf_data2[i].clone().unsqueeze(0)
            conf_scores3 = conf_data3[i].clone().unsqueeze(0)
            conf_temp_1 = torch.cat([conf_scores1, conf_scores2, conf_scores3],0)
            # conf_scores1 = conf_data1[i].clone().unsqueeze(0)
            # conf_scores2 = conf_data2[i].clone().unsqueeze(0)
            # conf_scores3 = conf_data3[i].clone().unsqueeze(0)
            # conf_temp_2 = torch.cat([conf_scores1, conf_scores2, conf_scores3], 0)
            maxconf_in_each_anchor, idx = torch.max(conf_temp_1, 2)
            maxconf_in_each_anchor[idx.eq(0)] = 0
            _, maxconf_across_scale_idx = torch.max(maxconf_in_each_anchor, 0)
            for j in range(num_priors):
                scale_idx = maxconf_across_scale_idx[j]
                choose_result_conf[i][j] = conf_temp_1[scale_idx][j]
                choose_result_loc[i][j] = loc_temp[scale_idx][j]
        return choose_result_loc, choose_result_conf

    def donms(self, arm_loc, arm_conf, loc_data, conf_data, prior, nobj_idx, for_what):
        conf_data[nobj_idx.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        num_priors = prior.size(0)    # """way 3"""
        # num_priors = loc_data.size(1)   # """way 3"""
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            default = decode(arm_loc[i], prior, self.variance)
            default = center_size(default)
            # default = torch.cat([default, default, default], 0) # """way 3"""
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            # print(decoded_boxes, conf_scores)
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                # print(scores.dim())
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                """way 1"""
                # wh = boxes[:, 2:]-boxes[:, :2]
                # area = wh[:, 0]*wh[:, 1]
                # scale = torch.sqrt(area)
                #
                # index_for_middle_1 = scale < 0.3
                # index_for_middle_2 = scale > 0.6
                # index_for_small = scale < 0.3
                # index_for_big = scale > 0.6
                # index_for_middle = ~(index_for_middle_1 ^ index_for_middle_2) #
                # if for_what == 'small':
                #     boxes = boxes[index_for_small]
                #     scores = scores[index_for_small]
                # if for_what == 'middle':
                #     boxes = boxes[index_for_middle]
                #     scores = scores[index_for_middle]
                # if for_what == 'big':
                #     boxes = boxes[index_for_big]
                #     scores = scores[index_for_big]
                # if scores.size(0) == 0:
                #     continue
                # idx of highest scoring and non-overlapping boxes per class
                # print(boxes, scores)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

    def forward(self, arm_loc_data, arm_conf_data, trm_loc_data1, trm_conf_data1, trm_loc_data2, trm_conf_data2,
                trm_loc_data3, trm_conf_data3, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc_data1 = trm_loc_data1
        conf_data1 = trm_conf_data1
        loc_data2 = trm_loc_data2
        conf_data2 = trm_conf_data2
        loc_data3 = trm_loc_data3
        conf_data3 = trm_conf_data3

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_thre
        """way 3"""
        # new_loc = torch.cat([loc_data1, loc_data2, loc_data3], 1)
        # new_conf = torch.cat([conf_data1, conf_data2, conf_data3], 1)
        # new_no_object_index = torch.cat([no_object_index, no_object_index, no_object_index], 1)
        # result = self.donms(arm_loc_data, arm_conf_data, new_loc, new_conf, prior_data, new_no_object_index, for_what='')

        """way 2"""
        new_loc, new_conf = self.chooseMaxScore(loc_data1, conf_data1, loc_data2, conf_data2, loc_data3, conf_data3, prior_data, no_object_index)
        result = self.donms(arm_loc_data, arm_conf_data, new_loc, new_conf, prior_data, no_object_index, for_what='')

        """way 1"""
        # result = self.donms(arm_loc_data, arm_conf_data, loc_data1, conf_data1, prior_data, no_object_index, for_what='small')
        # result = self.donms(arm_loc_data, arm_conf_data, loc_data2, conf_data2, prior_data, no_object_index, for_what='middle')
        # result = self.donms(arm_loc_data, arm_conf_data, loc_data3, conf_data3, prior_data, no_object_index, for_what='big')
        # result = torch.cat([result1, result2, result3], 2)
        return result
