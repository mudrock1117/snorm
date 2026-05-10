import numpy as np
import random
from functools import partial
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

def get_IoU(bboxes, rect_bboxes, weight=None, eps=1e-6):
    target = bboxes 
    pred = rect_bboxes

    pn, tn = pred.size(0), target.size(0)
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_right - target_left) * (target_bottom - target_top)

    x0_intersect = torch.max(pred_left.unsqueeze(0).expand(tn, pn), target_left.unsqueeze(1).expand(tn, pn)) 
    x1_intersect = torch.min(pred_right.unsqueeze(0).expand(tn, pn), target_right.unsqueeze(1).expand(tn, pn)) 
    y0_intersect = torch.max(pred_top.unsqueeze(0).expand(tn, pn), target_top.unsqueeze(1).expand(tn, pn)) 
    y1_intersect = torch.min(pred_bottom.unsqueeze(0).expand(tn, pn), target_bottom.unsqueeze(1).expand(tn, pn)) 
    w_intersect = x1_intersect - x0_intersect
    h_intersect = y1_intersect - y0_intersect
    w_intersect[w_intersect<0] = 0
    h_intersect[h_intersect<0] = 0
    area_intersect = w_intersect * h_intersect
    
    x0_area = torch.min(pred_left.unsqueeze(0).expand(tn, pn), target_left.unsqueeze(1).expand(tn, pn)) 
    x1_area = torch.max(pred_right.unsqueeze(0).expand(tn, pn), target_right.unsqueeze(1).expand(tn, pn)) 
    y0_area = torch.min(pred_top.unsqueeze(0).expand(tn, pn), target_top.unsqueeze(1).expand(tn, pn)) 
    y1_area = torch.max(pred_bottom.unsqueeze(0).expand(tn, pn), target_bottom.unsqueeze(1).expand(tn, pn)) 
    w_area = x1_area - x0_area
    h_area = y1_area - y0_area
    w_area[w_area<0] = 0
    h_area[h_area<0] = 0
    s_area = w_area * h_area

    gious = area_intersect / (s_area + eps)

    #ious = (area_intersect + 1.0) / (area_union + 1.0)
    return gious
###########################################################
def generate_moe_mask_v4(images_set, targets, stride=16):
    #generate mask
    weights  = []
    with torch.no_grad():
        locations = compute_locations(images_set, fpn_strides=[1])
        for idx in range(len(images_set)):
            labels, _, scores_f = compute_targets_for_locations_v5(locations[idx], targets, images_set[idx], stride=stride)
            weights.append(labels)
    
    return weights

##########################Mehod 5##########################
def class_loss_l2(preds, masks):
    preds = preds.sigmoid()
    loss = (preds - masks) ** 2
    #print('---2:', loss.mean(), masks.max(), masks.min(), preds.max(), preds.min())
    #print(loss.mean(dim=(0,2,3)), loss.size())
    loss = loss.mean(dim=1).mean()
    return loss

def class_loss(preds, masks):
    
    loss = 0.0
    #print(preds.size(), masks[:, 0].size())
    loss = F.binary_cross_entropy_with_logits(
            preds, masks
        )
    #print(loss)
    return loss

def class_loss_v2(preds, masks):
    #print(preds.size(), masks.size())
    criterion_ce = nn.CrossEntropyLoss()
    loss_ce = criterion_ce(preds, masks[:,0].long())
    loss = loss_ce
    return loss


def generate_moe_mask_v5(images_set, targets, stride=16):
    #generate mask
    weights  = []
    with torch.no_grad():
        locations = compute_locations(images_set, fpn_strides=[1])
        for idx in range(len(images_set)):
            labels, _, scores_f = compute_targets_for_locations_v5(locations[idx], targets, images_set[idx], stride=stride)
            weights.append(scores_f)
    
    return weights

def compute_locations(features, fpn_strides=[8]):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locations_per_level = compute_locations_per_level(
            h, w, fpn_strides[level],
            feature.device
        )
        locations.append(locations_per_level)
    return locations

def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def compute_targets_for_locations_v5(locations, targets, feats, stride=16):
    bn, lc, lh, lw = feats.size()
    labels = []
    scores_f = []
    reg_targets = []
    xs, ys = locations[:, 0], locations[:, 1]
    for im_i in range(len(targets)):
        targets_per_im = targets[im_i]
        bboxes_ = targets_per_im['boxes']
        labels_per_im = targets_per_im['labels']
        
        bboxes_ = bboxes_ * 1.0 / stride
        cx = (bboxes_[:, 0] + bboxes_[:, 2]) / 2
        cy = (bboxes_[:, 1] + bboxes_[:, 3]) / 2

        cw = bboxes_[:, 2] - bboxes_[:, 0] + 4
        ch = bboxes_[:, 3] - bboxes_[:, 1] + 4

        cx0 = cx - cw / 2
        cx1 = cx + cw / 2
        cy0 = cy - ch / 2
        cy1 = cy + ch / 2

        bboxes = torch.stack([cx0, cy0, cx1, cy1], dim=1)

        if "gate_mask" in targets_per_im.keys():
            scores_f_per_im = targets_per_im['gate_mask']
            #print(scores_f_per_im)
        
        if bboxes.size(0) == 0:
            labels_per_im = torch.zeros_like(locations[:,0])
            scores_f_per_im = torch.zeros_like(locations[:,0])
            reg_targets_per_im = torch.zeros_like(torch.cat([locations, locations], dim=1))
        else:
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            locations_to_gt_area = torch.ones_like(max_reg_targets_per_im)#area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = 0

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.max(dim=1)
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == 0] = 0
            if "gate_mask" in targets_per_im.keys():
                scores_f_per_im = scores_f_per_im[locations_to_gt_inds]
                scores_f_per_im[locations_to_min_area == 0] = 0.0
                #print('---0:', scores_f_per_im.max())
            else:
                scores_f_per_im = torch.zeros_like(labels_per_im) 
            
        labels.append(labels_per_im)
        scores_f.append(scores_f_per_im)
        reg_targets.append(reg_targets_per_im)
    
    nbn = len(labels)
    labels = torch.stack(labels, dim=0)
    labels = labels.view(nbn, lh, lw)
    
    scores_f = torch.stack(scores_f, dim=0)
    #print('----0:', labels.size(),scores_f.size())
    scores_f = scores_f.view(nbn, lh, lw)
    #print('---4:', scores_f.max())
    
    #scores_f = (labels>0).float()
    #scores_f = (scores_f>0.4).float()
    
    reg_targets = torch.stack(reg_targets, dim=0)
    reg_targets = reg_targets.view(nbn, lh, lw, 4)
    
    return labels, reg_targets, scores_f


##########################Mehod 6##########################
def select_rect_weight_w_prob(images, targets_, pseudo_targets, iter_index=0, features=None, stride=16):
    #compute gate label
    bn = len(targets_)
    
    ltargets = []
    for idx_ in range(bn):
        labels = copy.deepcopy(targets_[idx_]['labels'])
        boxes = copy.deepcopy(targets_[idx_]['boxes'])
        #score = labels * 0.0

        ###compue score
        labels_ps = copy.deepcopy(pseudo_targets[idx_]['labels'])
        boxes_ps = copy.deepcopy(pseudo_targets[idx_]['boxes'])
        score_ps = copy.deepcopy(pseudo_targets[idx_]['scores'])
        #print('---1:', score_ps.max(), score_ps.min())
        if labels_ps.size(0) > 0:
            iou_Mat = get_IoU(boxes, boxes_ps)
            iou_Mat_v, iou_Mat_i = iou_Mat.max(dim=1)
            cls_Mat = labels[:, None] - labels_ps[None, :] == 0
            scr_Mat = torch.index_select(score_ps, 0, iou_Mat_i)
            cls_Matb = torch.gather(cls_Mat, 1, iou_Mat_i.unsqueeze(1))[:,0]

            mask_box = iou_Mat_v > 0.3
            mask_cls = cls_Matb
            mask_final = (mask_box & mask_cls).float()

            scr_Mat = scr_Mat * mask_final 
        else:
            scr_Mat = labels * 0.0
        ltargets.append({'labels':labels, "boxes":boxes, 'gate_mask':scr_Mat})

    
    gate_mask = generate_moe_mask_v5([features], ltargets, stride=stride)
    gate_mask = gate_mask[0]

    return gate_mask

def select_rect_weight_w_prob_v2(images, targets_, pseudo_targets, iter_index=0, features=None, stride=16):
    #compute gate label
    bn = len(targets_)
    
    ltargets = []
    for idx_ in range(bn):
        labels = copy.deepcopy(targets_[idx_]['labels'])
        boxes = copy.deepcopy(targets_[idx_]['boxes'])
        #score = labels * 0.0

        ###compue score
        labels_ps = copy.deepcopy(pseudo_targets[idx_]['labels'])
        boxes_ps = copy.deepcopy(pseudo_targets[idx_]['boxes'])
        score_ps = copy.deepcopy(pseudo_targets[idx_]['scores'])
        #print('---1:', score_ps.max(), score_ps.min())
        if labels_ps.size(0) > 0:
            iou_Mat = get_IoU(boxes, boxes_ps)
            iou_Mat_v, iou_Mat_i = iou_Mat.max(dim=1)
            cls_Mat = labels[:, None] - labels_ps[None, :] == 0
            scr_Mat = torch.index_select(score_ps, 0, iou_Mat_i)
            cls_Matb = torch.gather(cls_Mat, 1, iou_Mat_i.unsqueeze(1))[:,0]

            mask_box = iou_Mat_v 
            mask_cls = cls_Matb
            #mask_final = (mask_box & mask_cls).float()

            scr_Mat = scr_Mat *  mask_box * mask_cls.float()


        else:
            scr_Mat = labels * 0.0
        ltargets.append({'labels':labels, "boxes":boxes, 'gate_mask':scr_Mat})

    
    gate_mask = generate_moe_mask_v5([features], ltargets, stride=stride)
    gate_mask = gate_mask[0]

    return gate_mask


def select_rect_weight_w_prob_1030(images, targets_, pseudo_targets, iter_index=0, features=None, stride=16):
    #compute gate label
    bn = len(targets_)
    
    ltargets = []
    for idx_ in range(bn):
        labels = copy.deepcopy(targets_[idx_]['labels'])
        boxes = copy.deepcopy(targets_[idx_]['boxes'])
        #score = labels * 0.0

        ###compue score
        labels_ps = copy.deepcopy(pseudo_targets[idx_]['labels'])
        boxes_ps = copy.deepcopy(pseudo_targets[idx_]['boxes'])
        score_ps = copy.deepcopy(pseudo_targets[idx_]['scores'])
        #print('---1:', score_ps.max(), score_ps.min())
        if labels_ps.size(0) > 0:
            iou_Mat = get_IoU(boxes, boxes_ps)
            iou_Mat_v, iou_Mat_i = iou_Mat.max(dim=1)
            cls_Mat = labels[:, None] - labels_ps[None, :] == 0
            #score_Mat = score_ps.unsqueeze(0).expand(cls_Mat.size())
            scr_Mat = torch.index_select(score_ps, 0, iou_Mat_i)
            cls_Matb = torch.gather(cls_Mat, 1, iou_Mat_i.unsqueeze(1))[:,0]
            #cls_Matb = torch.gather(cls_Mat, 1, iou_Mat_i.unsqueeze(1))[:,0]

            mask_box = iou_Mat_v > 0.6
            mask_cls = cls_Matb
            mask_final = (mask_box & mask_cls).float()

            scr_Mat = scr_Mat * mask_final 
        else:
            scr_Mat = labels * 0.0
        ltargets.append({'labels':labels, "boxes":boxes, 'gate_mask':scr_Mat})

    
    gate_mask = generate_moe_mask_v5([features], ltargets, stride=stride)
    gate_mask = gate_mask[0]

    return gate_mask

########################################20251214###############################################
def class_hl_loss(x, masks):
    x = x.sigmoid()
    
    fg_mask = (masks > 0.6).float()
    bg_mask = 1 - fg_mask
    err_loss = (x - masks) ** 2
    err_loss_fg = (fg_mask * err_loss).sum() / (fg_mask.sum() + 1e-5)
    err_loss_bg = (bg_mask * err_loss).sum() / (bg_mask.sum() + 1e-5)

    err_loss = err_loss_fg + err_loss_fg
    return err_loss

