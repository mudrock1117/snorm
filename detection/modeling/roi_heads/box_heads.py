from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss
from detection.layers import cat
from detection.modeling.roi_heads.mask_head import ConvUpSampleMaskHead, mask_rcnn_loss
from detection.modeling.utils import BalancedPositiveNegativeSampler, BoxCoder, Matcher

def select_foreground_proposals(proposals, labels):
    fg_proposals = []
    fg_select_masks = []
    for i, (proposals_per_img, label_per_img) in enumerate(zip(proposals, labels)):
        fg_mask = label_per_img > 0
        fg_idxs = fg_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_img[fg_idxs])
        fg_select_masks.append(fg_mask)
    return fg_proposals, fg_select_masks


class VGG16BoxPredictor(nn.Module):
    def __init__(self, cfg, in_channels, dropout_i = True, is_eval_head=False, use_memory_conv=False):
        super().__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        pool_size = cfg.MODEL.ROI_BOX_HEAD.POOL_RESOLUTION

        self.extractor_c = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                ),
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                ),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1
            )
        )


        self.extractor_l = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
        )

        if  dropout_i:
            self.cls_score = nn.Sequential(
                            nn.Dropout(),
                            nn.Linear(in_channels, in_channels),
                            nn.ReLU(True),
                            nn.Linear(in_channels, num_classes)
                            )
            self.bbox_pred = nn.Sequential(
                            nn.Dropout(),
                            nn.Linear(in_channels, in_channels),
                            nn.ReLU(True),
                            nn.Linear(in_channels, num_classes * 4)
                            )
        else:
            self.cls_score = nn.Sequential(
                            nn.Linear(in_channels, in_channels),
                            nn.ReLU(True),
                            nn.Linear(in_channels, num_classes)
                            )
            self.bbox_pred = nn.Sequential(
                            nn.Linear(in_channels, in_channels),
                            nn.ReLU(True),
                            nn.Linear(in_channels, num_classes * 4)
                            )
        
    
    def forward(self, box_features, memory_index_cls=0, memory_index_inter=0, eval_mode=False):
        cbox_features = self.extractor_c(box_features)

        cbox_features = torch.mean(cbox_features, dim=(2, 3))
        lbox_features = self.extractor_l(cbox_features)
        class_logits = self.cls_score(lbox_features)
        box_regression = self.bbox_pred(lbox_features)

        return class_logits, box_regression, cbox_features
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
                #print(m.weight.data.mean())


class ResNetBoxPredictor(nn.Module):
    def __init__(self, cfg, in_channels, scale_w=None, dropout_p=None):
        super().__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        resnet_cls = models.resnet.__dict__[cfg.MODEL.BACKBONE.NAME](pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.extractor_cls = resnet_cls.layer4
        del resnet_cls

        resnet_regress = models.resnet.__dict__[cfg.MODEL.BACKBONE.NAME](pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.extractor_regress = resnet_regress.layer4
        del resnet_regress


        in_channels = self.extractor_cls[-1].conv3.out_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1
                )
        )

        self.regressor = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1
            )
        )

        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, box_features):
        ebox_features_cls = self.extractor_cls(box_features)
        #print(box_features.size(),ebox_features_cls.size())
        ebox_features_reg = self.extractor_regress(box_features)

        mbox_features_cls = self.classifier(ebox_features_cls)
        mbox_features_reg = self.regressor(ebox_features_reg)

        mbox_features_cls = torch.mean(mbox_features_cls, dim=(2, 3))
        mbox_features_reg = torch.mean(mbox_features_reg, dim=(2, 3))

        class_logits = self.cls_score(mbox_features_cls)
        box_regression = self.bbox_pred(mbox_features_reg)

        return class_logits, box_regression, box_features


BOX_PREDICTORS = {
    'vgg16_predictor': VGG16BoxPredictor,
    'resnet101_predictor': ResNetBoxPredictor,
    'resnet50_predictor': ResNetBoxPredictor,
}

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = cat(labels, dim=0)
    regression_targets = cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class BoxHead(nn.Module):
    def __init__(self, cfg, in_channels, dropout_i=True, is_eval_head=False, positive_fraction=0.25, BATCH_SIZE_PER_IMAGE=None, use_memory_conv=False):
        super().__init__()
        # fmt:off
        if BATCH_SIZE_PER_IMAGE is None:
            batch_size           = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        else:
            batch_size           = BATCH_SIZE_PER_IMAGE

        score_thresh         = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        nms_thresh           = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        detections_per_img   = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

        box_predictor        = cfg.MODEL.ROI_BOX_HEAD.BOX_PREDICTOR
        spatial_scale        = cfg.MODEL.ROI_BOX_HEAD.POOL_SPATIAL_SCALE
        pool_size            = cfg.MODEL.ROI_BOX_HEAD.POOL_RESOLUTION
        pool_type            = cfg.MODEL.ROI_BOX_HEAD.POOL_TYPE
        mask_on              = cfg.MODEL.MASK_ON

        # fmt:on

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.spatial_scale = spatial_scale
        self.mask_on = mask_on

        if pool_type == 'align':
            pooler = partial(ops.roi_align, output_size=(pool_size, pool_size), spatial_scale=spatial_scale, sampling_ratio=2)
        elif pool_type == 'pooling':
            pooler = partial(ops.roi_pool, output_size=(pool_size, pool_size), spatial_scale=spatial_scale)
        else:
            raise ValueError('Unknown pool type {}'.format(pool_type))
        self.pooler = pooler

        self.box_predictor = BOX_PREDICTORS[box_predictor](cfg, in_channels)
        self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.matcher = Matcher(0.5, 0.5, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size, positive_fraction, is_eval_head=is_eval_head)

        self.is_eval_head = is_eval_head

        if mask_on:
            self.mask_head = ConvUpSampleMaskHead(in_channels, num_classes=cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)
        
        #stastic the number of each class of samples
        self.eval_nms_thresh = 0.5
        self.eval_score_thresh = 0.6
        self.cfg = cfg


    def forward(self, features, proposals, img_metas, targets=None, results_parameters=None, eval_pms_mode=False, test_exm=0):
        if self.training and targets is not None and test_exm==0:
            #print('---3:', test_exm)
            with torch.no_grad():
                proposals, labels, regression_targets, masks = self.select_training_samples(proposals, targets)
 
        is_target_domain = self.training and targets is None
        roi_features = self.pooler(features, proposals)

        #class_logits, box_regression, box_features = self.box_predictor(roi_features)

        if is_target_domain:
            if results_parameters is not None:
                print('----4')
                class_logits, box_regression, box_features = self.box_predictor(roi_features)
                score_thresh = results_parameters['score_thresh']
                nms_thresh = results_parameters['nms_thresh']
                
                dets = self.post_processor(class_logits, box_regression, proposals, img_metas, score_thresh=score_thresh, nms_thresh=nms_thresh)

                eval_paramter_list = [0.0 for i in range(self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)]
                eval_paramter_list[-1] = 0.5
                dets_bb = self.target_post_processor(class_logits, box_regression, proposals, img_metas, eval_paramter_list=eval_paramter_list)
                scores_, labels_ = F.softmax(class_logits, dim=1).max(dim=1)
                pred_boxes = self.deconde_boxes(box_regression, proposals)

                img_metas_nw = self.get_mean_prob_per_cls(class_logits, box_regression, proposals, img_metas, labels_)

            else:
                class_logits, box_regression, box_features = self.box_predictor(roi_features)
                dets = self.post_processor(class_logits, box_regression, proposals, img_metas, score_thresh=0.6, nms_thresh=0.5)

                eval_paramter_list = [0.0 for i in range(self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)]
                eval_paramter_list[-1] = 0.5
                dets_bb = self.target_post_processor(class_logits, box_regression, proposals, img_metas, eval_paramter_list=eval_paramter_list)
                #dets_bb = self.post_processor(class_logits, box_regression, proposals, img_metas, score_thresh=0.4, nms_thresh=0.5)

                scores_, labels_ = F.softmax(class_logits, dim=1).max(dim=1)
                pred_boxes = self.deconde_boxes(box_regression, proposals)

                img_metas_nw = self.get_mean_prob_per_cls(class_logits, box_regression, proposals, img_metas, labels_)

            
            obj_preds = {'boxes':pred_boxes, 'scores': scores_, 'labels':labels_, 'proposals':torch.cat(proposals, dim=0), 'dets_bb':dets_bb, "img_metas":img_metas_nw}
            
            return dets, {}, proposals, box_features, roi_features, class_logits, None, obj_preds, box_regression

        if self.training and targets is not None:
            class_logits, box_regression, box_features = self.box_predictor(roi_features)
            classification_loss, box_loss = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

            loss = {
                'rcnn_cls_loss': classification_loss,
                'rcnn_reg_loss': box_loss,
            }
            if self.mask_on:
                mask_loss = self.forward_mask(features, proposals, masks, labels)
                loss.update(mask_loss)
            dets = []
            
            labels_ = torch.cat(labels, dim=0)
            bboxes_ = torch.cat(regression_targets, dim=0)
            pred_boxes = self.deconde_boxes(bboxes_, proposals)
            pred_boxes = pred_boxes.expand(pred_boxes.size(0), box_regression.size(1) // 4, pred_boxes.size(2))
            scores_ = torch.ones_like(labels_)
            img_metas_nw = self.get_mean_prob_per_cls(class_logits, box_regression, proposals, img_metas, labels_)
            dets = self.post_processor(class_logits, box_regression, proposals, img_metas)
            
            obj_preds = {'boxes':pred_boxes, 'scores': scores_, 'labels':labels_, 'proposals':torch.cat(proposals, dim=0), "img_metas":img_metas_nw}
            
        else:
            loss = {}
            labels = None
            if results_parameters is not None:
                class_logits, box_regression, box_features = self.box_predictor(roi_features)
                score_thresh = results_parameters['score_thresh']
                nms_thresh = results_parameters['nms_thresh']

                dets = self.post_processor(class_logits, box_regression, proposals, img_metas, score_thresh=score_thresh, nms_thresh=nms_thresh)
            else:
                class_logits, box_regression, box_features = self.box_predictor(roi_features)
                dets = self.post_processor(class_logits, box_regression, proposals, img_metas)
            
                
            if self.mask_on:
                dets = self.forward_mask(features, dets)
        
            scores_, labels_ = F.softmax(class_logits, dim=1).max(dim=1)
            pred_boxes = self.deconde_boxes(box_regression, proposals)
            obj_preds = {'boxes':pred_boxes, 'scores': scores_, 'labels':labels_, 'proposals':torch.cat(proposals, dim=0)}
        
        return dets, loss, proposals, box_features, roi_features, class_logits, labels, obj_preds, box_regression

    def forward_mask(self, features, proposals, masks=None, labels=None):
        if self.training:
            fg_proposals, fg_select_masks = select_foreground_proposals(proposals, labels)
            gt_masks = []
            fg_labels = []
            for m, masks_per_img, label_per_img in zip(fg_select_masks, masks, labels):
                gt_masks.append(masks_per_img[m])
                fg_labels.append(label_per_img[m])

            pooled_features = ops.roi_align(features, proposals,
                                            output_size=(14, 14),
                                            spatial_scale=self.spatial_scale,
                                            sampling_ratio=2)
            mask_features = pooled_features[cat(fg_select_masks, dim=0)]
            mask_logits = self.mask_head(mask_features)
            del pooled_features
            mask_loss = mask_rcnn_loss(mask_logits, gt_masks, fg_proposals, fg_labels)
            loss_dict = {'mask_loss': mask_loss}
            return loss_dict
        else:
            detections = proposals
            proposals = [det['boxes'] for det in detections]
            pooled_features = ops.roi_align(features, proposals,
                                            output_size=(14, 14),
                                            spatial_scale=self.spatial_scale,
                                            sampling_ratio=2)

            mask_logits = self.mask_head(pooled_features)
            detections = self.mask_inference(mask_logits, detections)
            return detections

    def mask_inference(self, pred_mask_logits, detections):
        # Select masks corresponding to the predicted classes
        num_boxes_per_image = [len(det['labels']) for det in detections]
        num_masks = pred_mask_logits.shape[0]
        assert sum(num_boxes_per_image) == num_masks

        class_pred = cat([det['labels'] for det in detections])
        indices = torch.arange(num_masks)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()

        # mask_probs_pred.shape: (B, 1, M, M)
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

        for prob, det in zip(mask_probs_pred, detections):
            det['masks'] = prob  # (N, 1, M, M)

        return detections
    
    def post_processor(self, class_logits, box_regression, proposals, img_metas, score_thresh=None, nms_thresh=None):
        num_classes = class_logits.shape[1]
        device = class_logits.device

        if box_regression.size(0) == 0:
            results = []
            for img_meta in img_metas:
                result = {
                    'boxes': torch.empty(0, 4, device=device, dtype=torch.float32),
                    'scores': torch.empty(0, device=device, dtype=torch.float32),
                    'labels': torch.empty(0, device=device, dtype=torch.int64),
                }
                results.append(result)
            return results

        boxes_per_image = [box.shape[0] for box in proposals]
        proposals = cat([box for box in proposals])
        pred_boxes = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), proposals
        )
        pred_boxes = pred_boxes.reshape(sum(boxes_per_image), -1, 4)

        pred_scores = F.softmax(class_logits, -1)
        
        if score_thresh is None:
            lscore_thresh = self.score_thresh
        else:
            lscore_thresh = score_thresh
        
        if nms_thresh is None:
            lnms_thresh = self.nms_thresh
        else:
            lnms_thresh = nms_thresh

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            pred_boxes = (pred_boxes,)
            pred_scores = (pred_scores,)
        else:
            pred_boxes = pred_boxes.split(boxes_per_image, dim=0)  # (N, #CLS, 4)
            pred_scores = pred_scores.split(boxes_per_image, dim=0)  # (N, #CLS)

        results = []
        for scores, boxes, img_meta in zip(pred_scores, pred_boxes, img_metas):
            width, height = img_meta['img_shape']
            boxes = box_ops.clip_boxes_to_image(boxes, (height, width))

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > lscore_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, lnms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            result = {
                'boxes': boxes.detach(),
                'scores': scores.detach(),
                'labels': labels.detach(),
            }
            #print(result['labels'])
            #print(result['boxes'])

            results.append(result)

        return results
    
    def post_processor_eval(self, class_logits, box_regression, proposals, img_metas, score_thresh=None, nms_thresh=None):
        num_classes = class_logits.shape[1]
        device = class_logits.device

        if box_regression.size(0) == 0:
            results = []
            for img_meta in img_metas:
                result = {
                    'boxes': torch.empty(0, 4, device=device, dtype=torch.float32),
                    'scores': torch.empty(0, device=device, dtype=torch.float32),
                    'labels': torch.empty(0, device=device, dtype=torch.int64),
                }
                results.append(result)
            return results

        boxes_per_image = [box.shape[0] for box in proposals]
        proposals = cat([box for box in proposals])
        pred_boxes = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), proposals
        )
        pred_boxes = pred_boxes.reshape(sum(boxes_per_image), -1, 4)

        pred_scores = F.softmax(class_logits, -1)
        
        if score_thresh is None:
            lscore_thresh = self.score_thresh
        else:
            lscore_thresh = score_thresh
        
        if nms_thresh is None:
            lnms_thresh = self.nms_thresh
        else:
            lnms_thresh = nms_thresh

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            pred_boxes = (pred_boxes,)
            pred_scores = (pred_scores,)
        else:
            pred_boxes = pred_boxes.split(boxes_per_image, dim=0)  # (N, #CLS, 4)
            pred_scores = pred_scores.split(boxes_per_image, dim=0)  # (N, #CLS)

        results = []
        for scores, boxes, img_meta in zip(pred_scores, pred_boxes, img_metas):
            width, height = img_meta['img_shape']
            boxes = box_ops.clip_boxes_to_image(boxes, (height, width))

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            #scores_v, scores_i = torch.topk(scores, 2, dim=1)
            #scores = scores - scores_v[:, 1:2]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)


            # remove low scoring boxes
            inds = torch.nonzero(scores > lscore_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, lnms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            result = {
                'boxes': boxes.detach(),
                'scores': scores.detach(),
                'labels': labels.detach(),
            }

            results.append(result)

        return results
    
    def target_post_processor(self, class_logits, box_regression, proposals, img_metas, score_thresh=1e-20, nms_thresh=0.5, eval_paramter_list=None, mode=0):
        num_classes = class_logits.shape[1]
        device = class_logits.device

        if box_regression.size(0) == 0:
            results = []
            for img_meta in img_metas:
                result = {
                    'boxes': torch.empty(0, 4, device=device, dtype=torch.float32),
                    'scores': torch.empty(0, device=device, dtype=torch.float32),
                    'labels': torch.empty(0, device=device, dtype=torch.int64),
                }
                results.append(result)
            return results

        boxes_per_image = [box.shape[0] for box in proposals]
        proposals = cat([box for box in proposals])
        pred_boxes = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), proposals
        )
        pred_boxes = pred_boxes.reshape(sum(boxes_per_image), -1, 4)

        pred_scores = F.softmax(class_logits, -1)
        #print(pred_scores)
        gl_pred_scores = pred_scores.clone().detach()
        pred_scores_v, pred_scores_i = torch.topk(pred_scores, 2, dim=1)
        pred_scores = pred_scores - pred_scores_v[:, 1:2]

        if eval_paramter_list is None:
            eval_paramter_list = self.eval_paramter_list

        bn, bd = pred_scores.size()
        tsr_total_socre_thresh = torch.zeros_like(pred_scores)
        tsr_score_thresh = torch.from_numpy(np.array(eval_paramter_list[0:bd-1])).to(pred_scores.device)
        tsr_score_thresh = tsr_score_thresh.view(1,-1)
        tsr_total_socre_thresh[:,1:bd] +=  tsr_score_thresh
        #if mode == 1:
            #print('-----el:', pred_scores.size(), eval_paramter_list)
        #    print('-----el:',tsr_total_socre_thresh)

        pred_scores = pred_scores - tsr_total_socre_thresh
        nms_thresh = eval_paramter_list[-1]
        #print(eval_paramter_list[0:bd-1], nms_thresh)

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            pred_boxes = (pred_boxes,)
            pred_scores = (pred_scores,)
            gl_pred_scores = (gl_pred_scores,)
        else:
            pred_boxes = pred_boxes.split(boxes_per_image, dim=0)  # (N, #CLS, 4)
            pred_scores = pred_scores.split(boxes_per_image, dim=0)  # (N, #CLS)
            gl_pred_scores = gl_pred_scores.split(boxes_per_image, dim=0)  # (N, #CLS)

        results = []
        for scores, boxes, gl_scores, img_meta in zip(pred_scores, pred_boxes, gl_pred_scores, img_metas):
            width, height = img_meta['img_shape']
            boxes = box_ops.clip_boxes_to_image(boxes, (height, width))

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
            
            gl_scores =  gl_scores[:,1:].mean(dim=0)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            result = {
                'boxes': boxes.detach(),
                'scores': scores.detach(),
                'labels': labels.detach(),
                #'gl_scores': gl_scores.cpu().numpy()
                'gl_scores': gl_scores.detach()
            }
            #if eval_paramter_list[-2] == 0.9:
            #    print(labels)
            #print(result['labels'])
            #print(result['boxes'])

            results.append(result)

        return results
    
    def get_mean_prob_per_cls(self, class_logits, box_regression, proposals, img_metas, labels=None):
        num_classes = class_logits.shape[1]
        device = class_logits.device

        boxes_per_image = [box.shape[0] for box in proposals]
        proposals = cat([box for box in proposals])
        pred_boxes = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), proposals
        )
        pred_boxes = pred_boxes.reshape(sum(boxes_per_image), -1, 4)

        pred_scores = F.softmax(class_logits, -1)
        pred_scores_v, pred_scores_i = torch.topk(pred_scores, 2, dim=1)
        pred_scores = pred_scores - pred_scores_v[:, 1:2]

        labels_ = torch.zeros_like(class_logits).scatter_(1, labels.unsqueeze(1), 1)

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            pred_boxes = (pred_boxes,)
            pred_scores = (pred_scores,)
            labels_ = (labels_,)
        else:
            pred_boxes = pred_boxes.split(boxes_per_image, dim=0)  # (N, #CLS, 4)
            pred_scores = pred_scores.split(boxes_per_image, dim=0)  # (N, #CLS)
            labels_ = labels_.split(boxes_per_image, dim=0)  # (N, #CLS)

        results = []
        probs_list = []
        for scores, boxes, img_meta, label_ in zip(pred_scores, pred_boxes, img_metas, labels_):
            #print(scores.size(), label_.size())
            prob_sc = scores.mul(label_).sum(dim=0)
            prob_sc = prob_sc / (label_.sum(dim=0) + 1e-20)
            #probs_list.append(prob_sc)
            img_meta['prob_sc'] = prob_sc
        
        return img_metas

    def select_training_samples(self, proposals, targets):
        labels = []
        regression_targets = []
        masks = []
        for batch_id in range(len(targets)):
            target = targets[batch_id]
            proposals_per_image = proposals[batch_id]

            match_quality_matrix = box_ops.box_iou(target['boxes'], proposals_per_image)
            matched_idxs = self.matcher(match_quality_matrix)

            matched_idxs_for_target = matched_idxs.clamp(0)

            target_boxes = target['boxes'][matched_idxs_for_target]
            target_labels = target['labels'][matched_idxs_for_target]

            if 'masks' in target:
                target_masks = target['masks'][matched_idxs_for_target]
                masks.append(target_masks)

            labels_per_image = target_labels.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                target_boxes, proposals_per_image
            )
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = list(proposals)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals[img_idx] = proposals[img_idx][img_sampled_inds]
            labels[img_idx] = labels[img_idx][img_sampled_inds]
            regression_targets[img_idx] = regression_targets[img_idx][img_sampled_inds]
            if len(masks) > 0:
                masks[img_idx] = masks[img_idx][img_sampled_inds]

        return proposals, labels, regression_targets, masks
    
    def deconde_boxes(self, box_regression, proposals):

        boxes_per_image = [box.shape[0] for box in proposals]
        total_boxes = sum(boxes_per_image)
        
        if total_boxes == 0:
            return torch.empty(0, 0, 4, device=box_regression.device, dtype=box_regression.dtype)
        
        proposals = cat([box for box in proposals])
        pred_boxes = self.box_coder.decode(
            box_regression.view(total_boxes, -1), proposals
        )
        
        pred_boxes = pred_boxes.reshape(total_boxes, -1, 4)
        return pred_boxes
    
    def set_thresh_parameters(self, results_parameters):
        #self.eval_nms_thresh = results_parameters['nms_thresh']
        #self.eval_score_thresh = results_parameters['score_thresh']

        self.eval_paramter_list = results_parameters
    
    def save_old_thresh_parameters(self, ):
        '''
        self.eval_old_nms_thresh = self.eval_nms_thresh
        self.eval_old_score_thresh = self.eval_score_thresh
        '''

        self.eval_old_paramter_list = self.eval_paramter_list
    
    def restore_thresh_parameters(self, ):
        '''
        self.eval_nms_thresh = self.eval_old_nms_thresh
        self.eval_score_thresh = self.eval_old_score_thresh
        '''
        self.eval_paramter_list = self.eval_old_paramter_list
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
                #print(m.weight.data.mean())
