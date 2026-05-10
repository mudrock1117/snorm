import numpy as np
import random
from functools import partial
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F


def generate_rect_boxes(tc_images):
    bn, bd, bh, bw = tc_images.size()
    nh = 3
    nw = 5
    nh_step = bh / nh
    nw_step = bw / nw

    loc_h = random.randint(0, nh)
    loc_w = random.randint(0, nw)

    x0 = loc_w * nw_step
    x1 = (loc_w + 1) * nw_step
    y0 = loc_h * nh_step
    y1 = (loc_h + 1) * nh_step

    rect_boxes = torch.from_numpy(np.array([x0,y0,x1,y1])).to(tc_images.device).float()
    rect_boxes = rect_boxes.view(1, -1).expand(bn, 4)

    return rect_boxes

def select_rect_proposals(targets, rect_bboxes, iou_t=0.8):
    bn = len(targets)
    rtargets = []
    for idx_ in range(bn):
        bboxes = targets[idx_]['boxes']
        overlap_v = get_IoU(bboxes, rect_bboxes)
        masks = overlap_v > iou_t

        bboxes = bboxes[masks]
        labels = targets[idx_]['labels'][masks]
        rtargets.append({'labels':labels, 'boxes':bboxes})
    
    return rtargets

####################001####################
def generate_rect_boxes_v2(tc_images):
    bn, bd, bh, bw = tc_images.size()
    nh = 1
    nw = 3
    nh_step = bh * 1.0 / nh
    nw_step = bw * 1.0 / nw
    device = tc_images.device

    shifts_x0 = torch.arange(
        0, nw, step=1,
        dtype=torch.float32, device=device
    )
    shifts_x1 = torch.arange(
        1, nw+1, step=1,
        dtype=torch.float32, device=device
    )
    shifts_y0 = torch.arange(
        0, nh, step=1,
        dtype=torch.float32, device=device
    )
    shifts_y1 = torch.arange(
        1, nh+1, step=1,
        dtype=torch.float32, device=device
    )
    shifts_x0 = shifts_x0 * nw_step
    shifts_x1 = shifts_x1 * nw_step
    shifts_y0 = shifts_y0 * nh_step
    shifts_y1 = shifts_y1 * nh_step
    shifts_x1[shifts_x1>bw] = bw
    shifts_y1[shifts_y1>bh] = bh
    shift_y0, shift_x0 = torch.meshgrid(shifts_y0, shifts_x0)
    shift_y1, shift_x1 = torch.meshgrid(shifts_y1, shifts_x1)
    shift_y0 = shift_y0.reshape(-1)
    shift_x0 = shift_x0.reshape(-1)
    shift_y1 = shift_y1.reshape(-1)
    shift_x1 = shift_x1.reshape(-1)
    rect_targets = torch.stack([shift_x0, shift_y0, shift_x1, shift_y1], dim=1)
    
    return rect_targets

def select_rect_proposals_v2(outputs_teacher, outputs_teacher_aug, targets, rect_bboxes, iou_t=0.8, stride=8):
    bn = len(targets)
    device = outputs_teacher['features'].device

    rtargets = []
    feats_naug_fused = []
    feats_naug_ml = []
    feats_aug_fused = []
    feats_aug_ml = []
    for idx_ in range(bn):
        bboxes = targets[idx_]['boxes']
        overlap_v = get_IoU(bboxes, rect_bboxes) #MxN

        '''
        overlap_vr = overlap_v.mean(dim=0)  #N
        overlap_vr_mask = (overlap_vr > 0.0).float()
        overlap_vr_prob = torch.rand(overlap_vr.size()).to(device) * overlap_vr_mask
        index_v, index_i = torch.topk(overlap_vr_prob, 1, dim=0)
        
        overlap_vn = get_IoU_inter(bboxes, rect_bboxes)
        overlap_vd = overlap_vn[:, index_i[0]]
        masks = overlap_vd > iou_t
        '''

        overlap_vn = get_IoU_inter(bboxes, rect_bboxes)
        masks = overlap_vn > iou_t
        masks_sm = (masks.sum(dim=0) > 0.0).float()
        overlap_vr_prob = torch.rand(masks_sm.size()).to(device) * masks_sm
        index_v, index_i = torch.topk(overlap_vr_prob, 1, dim=0)
        masks_f = masks[:, index_i[0]]
        #print(rect_bboxes[index_i[0]])

        
        bboxes = bboxes[masks_f]
        labels = targets[idx_]['labels'][masks_f]
        rtargets.append({'labels':labels, 'boxes':bboxes})

        #feature
        rect_bboxes_ = rect_bboxes / stride
        x0 = int(rect_bboxes_[index_i[0], 0].cpu().detach().numpy())
        y0 = int(rect_bboxes_[index_i[0], 1].cpu().detach().numpy())
        if idx_ == 0:
            x1 = int(rect_bboxes_[index_i[0], 2].cpu().detach().numpy())
            y1 = int(rect_bboxes_[index_i[0], 3].cpu().detach().numpy())
        else:
            x1 = x0 + feats_naug_fused[0].size(-1)
            y1 = y0 + feats_naug_fused[0].size(-2)
        #print('---0:', [x0, x1, y0, y1])
        #print(outputs_teacher['features'][idx_].size(), outputs_teacher['features'][idx_, :, y0:y1, x0:x1].size())
        feats_naug_fused.append(outputs_teacher['features'][idx_, :, y0:y1, x0:x1])
        if idx_ == 0:
            for idx_k in range(len(outputs_teacher['feat_dict'])):
                feats_naug_ml.append([outputs_teacher['feat_dict'][idx_k][idx_, :, y0:y1, x0:x1]])
        else:
            for idx_k in range(len(outputs_teacher['feat_dict'])):
                feats_naug_ml[idx_k].append(outputs_teacher['feat_dict'][idx_k][idx_, :, y0:y1, x0:x1])

        feats_aug_fused.append(outputs_teacher_aug['features'][idx_, :, y0:y1, x0:x1])
        if idx_ == 0:
            for idx_k in range(len(outputs_teacher_aug['feat_dict'])):
                feats_aug_ml.append([outputs_teacher_aug['feat_dict'][idx_k][idx_, :, y0:y1, x0:x1]])
        else:
            for idx_k in range(len(outputs_teacher_aug['feat_dict'])):
                feats_aug_ml[idx_k].append(outputs_teacher_aug['feat_dict'][idx_k][idx_, :, y0:y1, x0:x1])

    feats_naug_fused = torch.stack(feats_naug_fused, dim=0)
    feats_aug_fused = torch.stack(feats_aug_fused, dim=0)
    for idx_k in range(len(outputs_teacher['feat_dict'])):
        feats_naug_ml[idx_k] = torch.stack(feats_naug_ml[idx_k], dim=0)
        feats_aug_ml[idx_k] = torch.stack(feats_aug_ml[idx_k], dim=0)

    return rtargets, feats_naug_fused, feats_naug_ml, feats_aug_fused, feats_aug_ml

def select_rect_weight_v2(tc_wei_dict, tc_aug_wei_dict, targets, rect_bboxes, iou_t=0.8, stride=8):
    bn = len(targets)
    device = tc_wei_dict.device

    rtargets = []
    wei_naug_fused = []
    wei_aug_fused = []
    for idx_ in range(bn):
        bboxes = targets[idx_]['boxes']
        overlap_v = get_IoU(bboxes, rect_bboxes) #MxN

        '''
        overlap_vr = overlap_v.mean(dim=0)  #N
        overlap_vr_mask = (overlap_vr > 0.0).float()
        overlap_vr_prob = torch.rand(overlap_vr.size()).to(device) * overlap_vr_mask
        index_v, index_i = torch.topk(overlap_vr_prob, 1, dim=0)
        
        overlap_vn = get_IoU_inter(bboxes, rect_bboxes)
        overlap_vd = overlap_vn[:, index_i[0]]
        masks = overlap_vd > iou_t
        '''

        overlap_vn = get_IoU_inter(bboxes, rect_bboxes)
        masks = overlap_vn > iou_t
        masks_sm = (masks.sum(dim=0) > 0.0).float()
        overlap_vr_prob = torch.rand(masks_sm.size()).to(device) * masks_sm
        index_v, index_i = torch.topk(overlap_vr_prob, 1, dim=0)
        masks_f = masks[:, index_i[0]]
        #print(rect_bboxes[index_i[0]])

        
        bboxes = bboxes[masks_f]
        labels = targets[idx_]['labels'][masks_f]
        rtargets.append({'labels':labels, 'boxes':bboxes})

        #feature
        rect_bboxes_ = rect_bboxes / stride
        x0 = int(rect_bboxes_[index_i[0], 0].cpu().detach().numpy())
        y0 = int(rect_bboxes_[index_i[0], 1].cpu().detach().numpy())
        if idx_ == 0:
            x1 = int(rect_bboxes_[index_i[0], 2].cpu().detach().numpy())
            y1 = int(rect_bboxes_[index_i[0], 3].cpu().detach().numpy())
        else:
            x1 = x0 + wei_naug_fused[0].size(-1)
            y1 = y0 + wei_naug_fused[0].size(-2)
        #print('---0:', [x0, x1, y0, y1])
        #print(outputs_teacher['features'][idx_].size(), outputs_teacher['features'][idx_, :, y0:y1, x0:x1].size())
        #print(tc_wei_dict.size(), tc_wei_dict[idx_, :, y0:y1, x0:x1].size())
        wei_naug_fused.append(tc_wei_dict[idx_, :, y0:y1, x0:x1])
        wei_aug_fused.append(tc_aug_wei_dict[idx_, :, y0:y1, x0:x1])

    wei_naug_fused = torch.stack(wei_naug_fused, dim=0)
    wei_aug_fused = torch.stack(wei_aug_fused, dim=0)

    return rtargets, wei_naug_fused, wei_aug_fused

####################001####################
def get_IoU_inter(bboxes, rect_bboxes, weight=None, eps=1e-6):
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
    
    s_area = (target_right - target_left) * (target_bottom -target_top)
    #print(bboxes.size(), rect_bboxes.size(), area_intersect.size(), s_area.size())
    gious = area_intersect / (s_area[:, None] + eps)
    return gious


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


def select_best_feature(feat_naug, feat_s_naug, feat_aug, feat_s_aug, rect_targets, tc_targets, tc_aug_targets):
    kernels = len(tc_targets)
    bn = len(rect_targets)
    nfeat_naug = []
    nfeat_s_naug = []
    nfeat_aug = []
    nfeat_s_aug = []

    for idx_bn in range(bn):
        ###no_aug
        mAP_list = []
        for idx_k in range(kernels): 
            #print(rect_targets[idx_bn])
            true_boxes = rect_targets[idx_bn]['boxes']
            true_labels = rect_targets[idx_bn]['labels']
            det_labels = tc_targets[idx_k][idx_bn]['labels']
            det_boxes = tc_targets[idx_k][idx_bn]['boxes']
            det_scores = tc_targets[idx_k][idx_bn]['scores']
            true_difficulties = torch.zeros_like(true_labels)
            APs, mAP = calculate_mAP([det_boxes], [det_labels], [det_scores], [true_boxes], [true_labels], [true_difficulties])
            mAP_list.append(mAP)
        
        import heapq
        '''
        mAP_list = torch.cat(mAP_list, dim=0)
        mAP_v, mAP_i = torch.topk(mAP_list, 1, dim=0)
        nfeat_naug.append(feat_naug[idx_bn])
        nfeat_s_naug.append(feat_s_naug[mAP_i[0]][idx_bn])
        '''
        #print(mAP_list)
        mAP_i = heapq.nlargest(1, range(len(mAP_list)), mAP_list.__getitem__)
        nfeat_naug.append(feat_naug[idx_bn])
        nfeat_s_naug.append(feat_s_naug[mAP_i[0]][idx_bn])

        ##aug
        mAP_list_aug = []
        for idx_k in range(kernels): 
            true_boxes = rect_targets[idx_bn]['boxes']
            true_labels = rect_targets[idx_bn]['labels']
            det_labels = tc_aug_targets[idx_k][idx_bn]['labels']
            det_boxes = tc_aug_targets[idx_k][idx_bn]['boxes']
            det_scores = tc_aug_targets[idx_k][idx_bn]['scores']
            true_difficulties = torch.zeros_like(true_labels)
            APs, mAP = calculate_mAP([det_boxes], [det_labels], [det_scores], [true_boxes], [true_labels], [true_difficulties])
            mAP_list_aug.append(mAP)
        
        '''
        mAP_list_aug = torch.cat(mAP_list_aug, dim=0)
        mAP_aug_v, mAP_aug_i = torch.topk(mAP_list_aug, 1, dim=0)
        nfeat_aug.append(feat_aug[idx_bn])
        nfeat_s_aug.append(feat_s_aug[mAP_aug_i[0]][idx_bn])
        '''
        mAP_aug_i = heapq.nlargest(1, range(len(mAP_list_aug)), mAP_list_aug.__getitem__)
        nfeat_aug.append(feat_aug[idx_bn])
        nfeat_s_aug.append(feat_s_aug[mAP_aug_i[0]][idx_bn])
    
    nfeat_naug = torch.stack(nfeat_naug, dim=0)
    nfeat_s_naug = torch.stack(nfeat_s_naug, dim=0)
    nfeat_aug = torch.stack(nfeat_aug, dim=0)
    nfeat_s_aug = torch.stack(nfeat_s_aug, dim=0)
    return nfeat_naug, nfeat_s_naug, nfeat_aug, nfeat_s_aug

def select_best_weight(rect_targets, tc_targets, tc_aug_targets, writer=None):
    kernels = len(tc_targets)
    bn = len(rect_targets)
    wei_naug_labels = []
    wei_aug_labels = []
    device = rect_targets[0]['boxes'].device

    for idx_bn in range(bn):
        ###no_aug
        mAP_list = []
        for idx_k in range(kernels): 
            #print(rect_targets[idx_bn])
            true_boxes = rect_targets[idx_bn]['boxes']
            true_labels = rect_targets[idx_bn]['labels']
            det_labels = tc_targets[idx_k][idx_bn]['labels']
            det_boxes = tc_targets[idx_k][idx_bn]['boxes']
            det_scores = tc_targets[idx_k][idx_bn]['scores']
            true_difficulties = torch.zeros_like(true_labels)
            APs, mAP = calculate_mAP([det_boxes], [det_labels], [det_scores], [true_boxes], [true_labels], [true_difficulties])
            mAP_list.append(mAP)
            #if idx_bn == 0:
            #    writer.add_scalar("mAP_"+str(idx_k), mAP)
        
        #print(mAP_list)
        out_mAP_list = mAP_list
        mAP_list = torch.from_numpy(np.array(mAP_list)).to(device)
        mAP_v, mAP_i = torch.topk(mAP_list, 1, dim=0)
        naug_labels_one = F.one_hot(mAP_i, num_classes=kernels)
        wei_naug_labels.append(naug_labels_one)


        ##aug
        mAP_list_aug = []
        for idx_k in range(kernels): 
            true_boxes = rect_targets[idx_bn]['boxes']
            true_labels = rect_targets[idx_bn]['labels']
            det_labels = tc_aug_targets[idx_k][idx_bn]['labels']
            det_boxes = tc_aug_targets[idx_k][idx_bn]['boxes']
            det_scores = tc_aug_targets[idx_k][idx_bn]['scores']
            true_difficulties = torch.zeros_like(true_labels)
            APs, mAP = calculate_mAP([det_boxes], [det_labels], [det_scores], [true_boxes], [true_labels], [true_difficulties])
            mAP_list_aug.append(mAP)
            #if idx_bn == 0:
            #    writer.add_scalar("mAP_aug_"+str(idx_k), mAP)
        
        out_mAP_list_aug = mAP_list_aug
        mAP_list_aug = torch.from_numpy(np.array(mAP_list_aug)).to(device)
        mAP_aug_v, mAP_aug_i = torch.topk(mAP_list_aug, 1, dim=0)
        aug_labels_one = F.one_hot(mAP_aug_i, num_classes=kernels)
        wei_aug_labels.append(aug_labels_one)
    
    wei_naug_labels = torch.cat(wei_naug_labels, dim=0)
    wei_aug_labels = torch.cat(wei_aug_labels, dim=0)
    
    #print(wei_naug_labels.size(), wei_aug_labels.size())
    return wei_naug_labels, wei_aug_labels, out_mAP_list, out_mAP_list_aug


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = 9 #len(label_map)
    device = det_boxes[0].device

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects is the total no. of objects across all images
    #print(det_boxes[0].size(), det_labels[0].size(), det_scores[0].size(), true_boxes[0].size(), true_labels[0].size(), true_difficulties[0].size())
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)
    
    #print('---:', det_images.size() , det_boxes.size() , det_labels.size() , det_scores.size())
    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    #average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    #return average_precisions, mean_average_precision
    return None, mean_average_precision



##################################
def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


##########################Mehod 3##########################
def select_rect_weight_v3(images, targets_, tc_targets, tc_aug_targets, iter_index=0, features=None, stride=16):
    #compute gate label
    bn = len(targets_)
    kernel_num = len(tc_targets)
    ntargets = []
    ntargets_aug = []
    for idx_ in range(bn):
        labels = targets_[idx_]['labels']
        boxes = targets_[idx_]['boxes']
        #print('---0:', tc_targets[0][idx_]['boxes'])
        #print('---1:', tc_targets[1][idx_]['boxes'])
        #non aug
        if tc_targets is not None:
            score_set = []
            for idx_k in range(kernel_num):
                labels_naug = tc_targets[idx_k][idx_]['labels']
                boxes_naug = tc_targets[idx_k][idx_]['boxes']
                preds_naug = tc_targets[idx_k][idx_]['scores']
                
                if labels_naug.size(0) > 0:
                    label_mask = ((labels[:, None] - labels_naug[None, :])==0).float()
                    naug_iou = get_IoU(boxes, boxes_naug)  * preds_naug[None, :]
                    topk_num = min(1, labels_naug.size(0))
                    naug_iou_v, naug_iou_i = torch.topk(naug_iou * label_mask, topk_num, dim=1)
                    score = naug_iou_v.mean(dim=1) #m
                else:
                    score = labels * 0.0
                
                score_set.append(score)
            
            score_set = torch.stack(score_set, dim=1) #m, kernel_num
            #print(score_set)
            #if iter_index % 200 ==0 and idx_ == 0:
            #    print(score_set)
            score_v, score_i = torch.topk(score_set, 1, dim=1) #m
            #print(score_i)
            #if (score_i>1).float().sum() > 20:
            #    print('----0:', score_v)
            #print((score_i==3).float().sum(), score_i.size())
            ignore_mask = (score_v == 0).long()
            score_i = score_i * (1 - ignore_mask) -1 * ignore_mask
            target = {'labels':labels, "boxes":boxes, 'gate_mask':score_i[:, 0]}
            ntargets.append(target)

        #aug
        if tc_aug_targets is not None:
            score_aug_set = []
            for idx_k in range(kernel_num):
                labels_aug = tc_aug_targets[idx_k][idx_]['labels']
                boxes_aug = tc_aug_targets[idx_k][idx_]['boxes']
                preds_aug = tc_aug_targets[idx_k][idx_]['scores']
                
                if labels_aug.size(0) > 0:
                    label_mask = ((labels[:, None] - labels_aug[None, :])==0).float()
                    aug_iou = get_IoU(boxes, boxes_aug)  * preds_aug
                    topk_num = min(1, labels_aug.size(0))
                    aug_iou_v, aug_iou_i = torch.topk(aug_iou * label_mask, topk_num, dim=1)
                    score = aug_iou_v.mean(dim=1) #m
                else:
                    score = labels * 0.0
                score_aug_set.append(score)
            
            score_aug_set = torch.stack(score_aug_set, dim=1) #m, kernel_num
            #print(score_aug_set)
            if iter_index % 200 ==0 and idx_ == 0:
                print(score_aug_set)
            score_aug_v, score_aug_i = torch.topk(score_aug_set, 1, dim=1) #m
            ignore_aug_mask = (score_aug_v == 0).long()
            score_aug_i = score_aug_i * (1 - ignore_aug_mask) -1 * ignore_aug_mask
            #print('---1:', labels.size(), score_aug_i[:, 0].size())
            target_aug = {'labels':labels, "boxes":boxes, 'gate_mask':score_aug_i[:, 0]}
            ntargets_aug.append(target_aug)
    
    gate_mask = None
    if tc_targets is not None:
        gate_mask = generate_moe_mask([features], ntargets, stride=stride)
        gate_mask = gate_mask[0]
    
    
    gate_aug_mask = None
    if tc_aug_targets is not None:
        gate_aug_mask = generate_moe_mask([features], ntargets_aug, stride=stride)
        gate_aug_mask = gate_aug_mask[0]

    return gate_mask, gate_aug_mask

def generate_moe_mask(images_set, targets, stride=16):
    #generate mask
    weights  = []
    with torch.no_grad():
        locations = compute_locations(images_set, fpn_strides=[1])
        for idx in range(len(images_set)):
            labels, _, scores_f = compute_targets_for_locations(locations[idx], targets, images_set[idx], stride=stride)
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

def compute_targets_for_locations(locations, targets, feats, stride=16):
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
                scores_f_per_im[locations_to_min_area == 0] = -1
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
    
    reg_targets = torch.stack(reg_targets, dim=0)
    reg_targets = reg_targets.view(nbn, lh, lw, 4)
    
    return labels, reg_targets, scores_f

def get_IoU(target, pred, weight=None, eps=1e-6):
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


##########################Mehod 4##########################
def select_rect_weight_v4(images, targets_, tc_targets, tc_aug_targets, iter_index=0, features=None, stride=16):
    #compute gate label
    bn = len(targets_)
    kernel_num = len(tc_targets)
    ntargets = []
    ntargets_aug = []

    #non aug
    gate_masks = []
    if tc_targets is not None:
        score_set = []
        for idx_k in range(kernel_num):
            
            ltargets = []
            for idx_ in range(bn):
                labels = copy.deepcopy(targets_[idx_]['labels'])
                boxes = copy.deepcopy(targets_[idx_]['boxes'])

                labels_naug = tc_targets[idx_k][idx_]['labels']
                boxes_naug = tc_targets[idx_k][idx_]['boxes']
                preds_naug = tc_targets[idx_k][idx_]['scores']

                if labels_naug.size(0) > 0:
                    label_mask = ((labels[:, None] - labels_naug[None, :])==0).float()
                    naug_iou = get_IoU(boxes, boxes_naug)  * preds_naug[None, :]
                    topk_num = min(1, labels_naug.size(0))
                    naug_iou_v, naug_iou_i = torch.topk(naug_iou * label_mask, topk_num, dim=1)
                    score = naug_iou_v.mean(dim=1) #m
                else:
                    score = labels * 0.0
                
                ltargets.append({'labels':labels, "boxes":boxes, 'gate_mask':score})

            
            gate_mask = generate_moe_mask_v4([features], ltargets, stride=stride)
            gate_mask = gate_mask[0]
            gate_masks.append(gate_mask)
        

    #aug
    gate_aug_masks = []
    if tc_aug_targets is not None:
        for idx_k in range(kernel_num):
            
            mtargets = []
            for idx_ in range(bn):
                labels = copy.deepcopy(targets_[idx_]['labels'])
                boxes = copy.deepcopy(targets_[idx_]['boxes'])

                labels_aug = tc_aug_targets[idx_k][idx_]['labels']
                boxes_aug = tc_aug_targets[idx_k][idx_]['boxes']
                preds_aug = tc_aug_targets[idx_k][idx_]['scores']
                
                if labels_aug.size(0) > 0:
                    label_mask = ((labels[:, None] - labels_aug[None, :])==0).float()
                    aug_iou = get_IoU(boxes, boxes_aug)  * preds_aug
                    topk_num = min(1, labels_aug.size(0))
                    aug_iou_v, aug_iou_i = torch.topk(aug_iou * label_mask, topk_num, dim=1)
                    score = aug_iou_v.mean(dim=1) #m
                else:
                    score = labels * 0.0
                #print(score)
                mtargets.append({'labels':labels, "boxes":boxes, 'gate_mask':score})
        
            gate_aug_mask = generate_moe_mask_v4([features], mtargets, stride=stride)
            gate_aug_mask = gate_aug_mask[0]
            gate_aug_masks.append(gate_aug_mask)
            

    return gate_masks, gate_aug_masks

def scoring_loss(preds, masks):
    loss = 0.0
    for idx in range(len(preds)):
        mark = (masks[idx] >= 0.0).float()
        errors = (preds[idx] - masks[idx]) ** 2
        loss += (errors * mark).sum() / mark.sum()
    return loss


def generate_moe_mask_v4(images_set, targets, stride=16):
    #generate mask
    weights  = []
    with torch.no_grad():
        locations = compute_locations(images_set, fpn_strides=[1])
        for idx in range(len(images_set)):
            labels, _, scores_f = compute_targets_for_locations_v4(locations[idx], targets, images_set[idx], stride=stride)
            weights.append(scores_f)
    
    return weights

def compute_targets_for_locations_v4(locations, targets, feats, stride=16):
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
                scores_f_per_im[locations_to_min_area == 0] = -1
                #print('---0:', scores_f_per_im.max())
            else:
                scores_f_per_im = torch.zeros_like(labels_per_im) - 1
            
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
    
    reg_targets = torch.stack(reg_targets, dim=0)
    reg_targets = reg_targets.view(nbn, lh, lw, 4)
    
    return labels, reg_targets, scores_f


##########################Mehod 5##########################
def select_rect_weight_v5(images, targets_, tc_targets, tc_aug_targets, iter_index=0, features=None, stride=16):
    #compute gate label
    bn = len(targets_)
    if tc_targets is not None:
        kernel_num = len(tc_targets)
    else:
        kernel_num = len(tc_aug_targets)
    ntargets = []
    ntargets_aug = []

    #non aug
    gate_masks = []
    if tc_targets is not None:
        score_set = []
        for idx_k in range(kernel_num):
            
            ltargets = []
            for idx_ in range(bn):
                labels = copy.deepcopy(targets_[idx_]['labels'])
                boxes = copy.deepcopy(targets_[idx_]['boxes'])

                labels_naug = tc_targets[idx_k][idx_]['labels']
                boxes_naug = tc_targets[idx_k][idx_]['boxes']
                preds_naug = tc_targets[idx_k][idx_]['scores']

                if labels_naug.size(0) > 0:
                    label_mask = ((labels[:, None] - labels_naug[None, :])==0).float()
                    naug_iou = get_IoU(boxes, boxes_naug)  * preds_naug[None, :]
                    topk_num = min(1, labels_naug.size(0))
                    naug_iou_v, naug_iou_i = torch.topk(naug_iou * label_mask, topk_num, dim=1)
                    score = naug_iou_v.mean(dim=1) #m
                else:
                    score = labels * 0.0
                
                ltargets.append({'labels':labels, "boxes":boxes, 'gate_mask':score})

            
            gate_mask = generate_moe_mask_v5([features], ltargets, stride=stride)
            gate_mask = gate_mask[0]
            gate_masks.append(gate_mask)
        

    #aug
    gate_aug_masks = []
    if tc_aug_targets is not None:
        for idx_k in range(kernel_num):
            
            mtargets = []
            for idx_ in range(bn):
                labels = copy.deepcopy(targets_[idx_]['labels'])
                boxes = copy.deepcopy(targets_[idx_]['boxes'])

                labels_aug = tc_aug_targets[idx_k][idx_]['labels']
                boxes_aug = tc_aug_targets[idx_k][idx_]['boxes']
                preds_aug = tc_aug_targets[idx_k][idx_]['scores']
                
                if labels_aug.size(0) > 0:
                    label_mask = ((labels[:, None] - labels_aug[None, :])==0).float()
                    aug_iou = get_IoU(boxes, boxes_aug)  * preds_aug
                    topk_num = min(1, labels_aug.size(0))
                    aug_iou_v, aug_iou_i = torch.topk(aug_iou * label_mask, topk_num, dim=1)
                    score = aug_iou_v.mean(dim=1) #m
                else:
                    score = labels * 0.0
                #print(score)
                mtargets.append({'labels':labels, "boxes":boxes, 'gate_mask':score})
        
            gate_aug_mask = generate_moe_mask_v5([features], mtargets, stride=stride)
            gate_aug_mask = gate_aug_mask[0]
            gate_aug_masks.append(gate_aug_mask)
            

    return gate_masks, gate_aug_masks

def class_loss(preds, masks):
    
    loss = 0.0
    for idx in range(len(preds)):
        #print(preds[idx].size(), masks[idx].size())
        #print(preds[idx].min(), preds[idx].max(), masks[idx].min(), masks[idx].max())
        loss += F.binary_cross_entropy_with_logits(
                preds[idx][:,0], masks[idx]
            )
    
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
                scores_f_per_im[locations_to_min_area == 0] = -1
                #print('---0:', scores_f_per_im.max())
            else:
                scores_f_per_im = torch.zeros_like(labels_per_im) - 1
            
        labels.append(labels_per_im)
        scores_f.append(scores_f_per_im)
        reg_targets.append(reg_targets_per_im)
    
    nbn = len(labels)
    labels = torch.stack(labels, dim=0)
    labels = labels.view(nbn, lh, lw)
    
    #scores_f = torch.stack(scores_f, dim=0)
    #print('----0:', labels.size(),scores_f.size())
    #scores_f = scores_f.view(nbn, lh, lw)
    #print('---4:', scores_f.max())
    
    scores_f = (labels>0).float()
    
    reg_targets = torch.stack(reg_targets, dim=0)
    reg_targets = reg_targets.view(nbn, lh, lw, 4)
    
    return labels, reg_targets, scores_f


##########################Mehod 6##########################
def select_rect_weight_v6(images, targets_, iter_index=0, features=None, stride=16):
    #compute gate label
    bn = len(targets_)
    
    ltargets = []
    for idx_ in range(bn):
        labels = copy.deepcopy(targets_[idx_]['labels'])
        boxes = copy.deepcopy(targets_[idx_]['boxes'])
        score = labels * 0.0
        
        ltargets.append({'labels':labels, "boxes":boxes, 'gate_mask':score})

    
    gate_mask = generate_moe_mask_v5([features], ltargets, stride=stride)
    gate_mask = gate_mask[0]
    

    return gate_mask

def select_rect_weight_v7(images, targets_, iter_index=0, features=None, stride=16, win_sz_set=None):
    #compute gate label
    bn = len(targets_)
    gate_mask_set = []
    for idx_w in range(len(win_sz_set)):
        win_sz = win_sz_set[idx_w]
        rect_boxes = generate_rect_boxes_v7(images, win_sz=win_sz)
        
        ltargets = []
        for idx_ in range(bn):
            labels = copy.deepcopy(targets_[idx_]['labels'])
            boxes = copy.deepcopy(targets_[idx_]['boxes'])
            score = labels * 0.0

            rect_box = rect_boxes #[idx_]
            over_mat = get_IoU_inter(boxes, rect_box)
            over_mat_v, over_mat_i = over_mat.max(dim=1)
            mask = over_mat_v > 0.9
            labels = labels[mask]
            boxes = boxes[mask]
            score = score[mask]

            ltargets.append({'labels':labels, "boxes":boxes, 'gate_mask':score})

        
        gate_mask = generate_moe_mask_v5([features], ltargets, stride=stride)
        gate_mask = gate_mask[0]
        gate_mask_set.append(gate_mask)
    

    return gate_mask_set

def generate_rect_boxes_v7(tc_images, win_sz=None):
    bn, bd, bh, bw = tc_images.size()
    nh = win_sz[0]
    nw = win_sz[1]
    nh_step = bh / nh
    nw_step = bw / nw

    device = tc_images.device

    shifts_y0 = torch.arange(
        0, nh * nh_step, step=nh_step,
        dtype=torch.float32, device=device
    )
    shifts_y1 = torch.arange(
        nh_step, (nh+1) * nh_step, step=nh_step,
        dtype=torch.float32, device=device
    )

    shifts_x0 = torch.arange(
        0, nw * nw_step, step=nw_step,
        dtype=torch.float32, device=device
    )
    shifts_x1 = torch.arange(
        nw_step, (nw+1) * nw_step, step=nw_step,
        dtype=torch.float32, device=device
    )

    shift_y0, shift_x0 = torch.meshgrid(shifts_y0, shifts_x0)
    shift_y1, shift_x1 = torch.meshgrid(shifts_y1, shifts_x1)

    shift_y0 = shift_y0.reshape(-1)
    shift_x0 = shift_x0.reshape(-1)
    shift_x1 = shift_x1.reshape(-1)
    shift_y1 = shift_y1.reshape(-1)

    #rect_boxes = torch.from_numpy(np.array([x0,y0,x1,y1])).to(tc_images.device).float()
    rect_boxes = torch.stack([shift_x0,shift_y0,shift_x1,shift_y1], dim=1).to(tc_images.device).float()

    return rect_boxes

#images, targets, img_metas
def select_rect_labels_v7(images, targets_, img_metas, features, win_sz_set=None):
    #compute gate label
    bn = len(targets_)
    

    win_sz = win_sz_set#[idx_w]
    rect_boxes = generate_rect_boxes_v7(images, win_sz=win_sz)
    nimages = []
    nimg_metas = []
    ltargets = []
    nfeatures = []
    for idx_ in range(bn):
        labels = copy.deepcopy(targets_[idx_]['labels'])
        boxes = copy.deepcopy(targets_[idx_]['boxes'])
        score = labels * 0.0

        rect_box = rect_boxes #[idx_]
        over_mat = get_IoU_inter(boxes, rect_box)
        over_mat_v, over_mat_i = over_mat.max(dim=1)
        mask = over_mat_v > 0.9
        labels = labels[mask]
        boxes = boxes[mask]
        score = score[mask]
        if labels.size(0) > 0:
            nimages.append(images[idx_])
            nimg_metas.append(img_metas[idx_])
            nfeatures.append(features[idx_])
            ltargets.append({'labels':labels, "boxes":boxes, 'gate_mask':score})

    
    if len(ltargets) > 0:
        nimages = torch.stack(nimages, dim=0)
        nfeatures = torch.stack(nfeatures, dim=0)

    return nimages, ltargets, nimg_metas, nfeatures