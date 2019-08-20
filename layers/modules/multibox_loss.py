# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp, decode, involve_min, get_smaller_size_matrix, get_futher_depth_matrix, \
    get_depth_difference, jaccard
from data import VOC_OBJ_MIN, VOC_OBJ_MAX, COCO_OBJ_MIN, COCO_OBJ_MAX


class MultiBoxLoss(nn.Module):
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
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
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
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t.requires_grad = False
        conf_t.requires_grad = False

        pos = conf_t > 0

        # ###################################################
        # # Depth estimation
        # ###################################################
        # db = 'VOC'
        # normalizer = 1.
        # loss_g = 0.
        # var_g = 1.
        # base_g = 2.
        # depth_data_size = depth_data.size(1)
        # loc_data_g = loc_data[:, :depth_data_size, :]
        # defaults_g = defaults[:depth_data_size, :]
        # defaults_g = defaults_g.unsqueeze(0).expand_as(loc_data_g)
        # # index per label
        # for n in range(1, self.num_classes):
        #     idx_class = conf_t == n
        #     idx_class = idx_class[:, :depth_data.size(1)]
        #     idx_class_g = idx_class.unsqueeze(idx_class.dim()).expand_as(depth_data)
        #     idx_class_l = idx_class.unsqueeze(idx_class.dim()).expand_as(loc_data_g)
        #     if db == 'VOC':
        #         bound = torch.tensor([VOC_OBJ_MIN[n] / normalizer, VOC_OBJ_MAX[n] / normalizer]).cuda()
        #     elif db == 'COCO':
        #         bound = torch.tensor([COCO_OBJ_MIN[n] / normalizer, COCO_OBJ_MAX[n] / normalizer]).cuda()
        #     batch_geo = torch.exp(depth_data[idx_class_g] * var_g) * base_g
        #     # Estimated box info.
        #     priors = defaults_g[idx_class_l].view(-1, 4)
        #     with torch.no_grad():
        #         locs = loc_data_g[idx_class_l].view(-1, 4)
        #         batch_est = priors[:, 2:] * torch.exp(locs[:, 2:] * self.variance[1])
        #         batch_est = torch.sqrt(batch_est[:, 0] * batch_est[:, 1])
        #     lower_bound = bound[0] - batch_geo * batch_est
        #     upper_bound = batch_geo * batch_est - bound[1]
        #     loss_g += torch.sum(F.softplus(lower_bound) + F.softplus(upper_bound))
        #     # num_geo += idx_class.long().sum(1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # # Geometric loss ver. 2
        # min_overlap = .8
        # pos_idx_g = pos.unsqueeze(pos.dim()).expand_as(depth_data)
        # num = loc_data.size(0)
        # M = 0.
        # loss_g2 = 0.
        # for n in range(num):
        #     decoded_boxes = decode(loc_data[n][pos_idx[n]].view(-1, 4), defaults[pos_idx[n]].view(-1, 4), self.variance)
        #     overlaps = involve_min(decoded_boxes, decoded_boxes)
        #     smaller_matrix = get_smaller_size_matrix(decoded_boxes)
        #     overlap_smaller_matrix = (overlaps > min_overlap) * smaller_matrix
        #
        #     depth = torch.exp(depth_data[n][pos_idx_g[n]] * var_g) * base_g
        #     further_depth_matrix = get_futher_depth_matrix(depth)
        #     d_index = overlap_smaller_matrix * further_depth_matrix
        #     depth_difference = get_depth_difference(depth)
        #     # Estimated box info.
        #     bound = depth_difference[d_index]
        #     M += torch.sum(d_index.float())
        #     loss_g2 += torch.sum(F.softplus(bound))
        #
        # loss_g2 /= max(M, 1.)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # loss_c = focal_loss(conf_p, targets_weighted)
        # loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        loss_c = custom_cross_entropy(conf_p, targets_weighted)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        # # Focal Loss
        # pos_neg = conf_t > -1
        # pos_neg_idx = pos_neg.unsqueeze(2).expand_as(conf_data)
        # conf_p = conf_data[pos_neg_idx.gt(0)].view(-1, self.num_classes)
        # targets_weighted = conf_t[pos_neg.gt(0)]
        # loss_c = focal_loss(conf_p, targets_weighted)

        num_pos = pos.long().sum(1, keepdim=True)
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


def custom_cross_entropy(pred, target):
    t = one_hot_embedding(target, pred.size(-1))
    t = label_smoothing(t, pred.size(-1))
    t = Variable(t).cuda()
    epsilon = 1e-7

    logit = F.softmax(pred, dim=1)
    logit = logit.clamp(epsilon, 1. - epsilon)
    logit = logit.log()
    conf_loss = - t.float() * logit

    return conf_loss.sum()


def focal_loss(pred, target):
    alpha = 0.25
    alpha_list = [1. - alpha]
    for n in range(pred.size(-1) - 1):
        alpha_list += [alpha]
    alpha = torch.Tensor(alpha_list).cuda()
    alpha = alpha.unsqueeze(0)
    gamma = 2.
    epsilon = 1e-7

    t = one_hot_embedding(target, pred.size(-1))
    # t = t[:, 1:]
    t = Variable(t).cuda()

    logit = F.softmax(pred, dim=1)
    logit = logit.clamp(epsilon, 1. - epsilon)
    loglogit = logit.log()
    powlogit = torch.pow(1. - logit, gamma)
    conf_loss = - alpha * loglogit * powlogit * t.float()
    conf_loss = conf_loss.sum()

    return conf_loss


def focal_loss_alt(pred, target):
    alpha = 0.25
    gamma = 2.

    t = one_hot_embedding(target, pred.size(-1))
    t = t[:, 1:]
    t = Variable(t).cuda()

    xt = pred * (2*t - 1)
    pt = (2 * xt + 1).sigmoid()

    w = alpha * t + (1 - alpha) * (1 - t)
    loss = -w * pt.log() / 2
    return loss.sum()


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]


def label_smoothing(one_hot_vec, num_classes):
    eps = 0.05
    one_hot_vec = one_hot_vec.float()
    one_hot_vec[one_hot_vec > 0.5] = 1. - eps
    one_hot_vec[one_hot_vec <= 0.5] = eps / (num_classes - 1)
    return one_hot_vec
