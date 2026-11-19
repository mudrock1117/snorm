import argparse
import datetime
import math
import os
import time
import copy
import random
import signal
import subprocess
import numpy as np

# CUDA预初始化：在任何其他导入之前初始化CUDA，防止tensorflow导入失败污染CUDA上下文
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 降低显存碎片化风险：限制单次分配上限，避免大块分裂
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
import torch
# Under launch, init_distributed_mode() must select LOCAL_RANK before any CUDA
# context is created. Otherwise every process first allocates on logical GPU 0.
if int(os.environ.get('WORLD_SIZE', '1')) == 1:
    torch.cuda.init()
    torch.cuda.synchronize()
import torchvision
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from detection import utils
from detection.config import cfg as det_cfg
from detection.data.build import build_data_loaders
from detection.data.source_diversification import SourceAdaptiveDiversification
from detection.engine.eval import evaluation
from detection.modeling.build import build_detectors
from detection.utils import dist_utils
from diagnostics.small_anchor import load_small_anchor_checkpoint
from diagnostics.b1_protected import (
    B1_OUTPUT_LR,
    audit_b1_trainable_parameters,
    configure_b1_trainable_parameters,
)
from diagnostics.b2_protected import (
    B2_OUTPUT_LR,
    audit_b2_trainable_parameters,
    configure_b2_trainable_parameters,
)
from diagnostics.a3_protected import (
    ProtectedSmallAnchorStep,
    A4_OUTPUT_LR,
    A4_SHARED_CONV_LR,
    audit_a3_trainable_parameters,
    audit_a4_trainable_parameters,
    configure_a3_trainable_parameters,
    configure_a4_trainable_parameters,
)

#from saver_domain import Saver
import torch.nn.functional as F
from detection.modeling.norm.object_aware_ahdr import masked_gate_statistics
#from pro_utils import write_img, get_params
#from functions import conduct_transformations


global_step = 0
total_steps = 0
best_mAP = -1.0
global_start = 0
global_index = -1
global_end = False
sample_index_iter = 0

global_step_gan = 0
is_update_memory = False

current_foggy_mAP = 0.0
current_night_mAP = 0.0
best_sum_mAP = 0.0
best_step = -1
best_mAP = 0.0

###memory module
def normalize_image(x):
        return x[:,0:3,:,:]
    
def assemble_outputs_test(images, t_images, images_sa2, images_sa3):
    images = normalize_image(images).detach()
    t_images = normalize_image(t_images).detach()
    images_set = [images, t_images]
    if images_sa2 is not None:
        images_sa2 = normalize_image(images_sa2).detach()
        images_set += [images_sa2]
    if images_sa3 is not None:
        images_sa3 = normalize_image(images_sa3).detach()
        images_set += [images_sa3]

    row1 = torch.cat(images_set,3)
    return row1
    
def write_img(images, t_images, images_sa2, images_sa3, ep, model, is_test=False, image_dir='./results/MGUIT/test'):
    fimage_dir = image_dir + '/sunny_night/images'
    if not is_test:
        assembled_images = assemble_outputs(images, t_images, images_sa2, images_sa3)
    else:
        assembled_images = assemble_outputs_test(images, t_images, images_sa2, images_sa3)
    img_filename = '%s/gen_%05d.jpg' % (fimage_dir, ep)
    torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)


def cosine_scheduler(eta_max, eta_min, current_step):
    y = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(current_step / total_steps * math.pi))
    return y


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def box_to_centers(boxes):
    x = boxes[:, 2] - boxes[:, 0]
    y = boxes[:, 3] - boxes[:, 1]
    centers = torch.stack((x, y), dim=1)
    return centers


def detach_features(features):
    if isinstance(features, torch.Tensor):
        return features.detach()
    return tuple([f.detach() for f in features])


def convert_sync_batchnorm(model):
    convert = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            convert = True
            break
    if convert:
        print('Convert to SyncBatchNorm')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def set_d1_frozen_detector_bn_eval(model):
    if os.environ.get('SNORM_EXPERIMENT', 'E0') != 'D1':
        return 0
    model_inner = model.module if hasattr(model, 'module') else model
    teacher = model_inner.teacher
    frozen_detector_modules = (
        teacher.backbone_0,
        teacher.backbone_1,
        teacher.backbone_2,
        teacher.backbone_3,
        teacher.rpn_norm,
        teacher.box_head_norm,
        teacher.rpn_eval,
        teacher.box_head_eval,
    )
    batch_norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    frozen_batch_norms = []
    for detector_module in frozen_detector_modules:
        for module in detector_module.modules():
            if isinstance(module, batch_norm_types):
                module.eval()
                frozen_batch_norms.append(module)
    training_batch_norms = [
        module for module in frozen_batch_norms if module.training
    ]
    if training_batch_norms:
        raise RuntimeError(
            '[D1] frozen detector BatchNorm/SyncBatchNorm remained in train mode'
        )
    if not teacher.proj_norm.render_net.training:
        raise RuntimeError('[D1] render_net must remain in train mode')
    return len(frozen_batch_norms)

def term_sig_hanlder(signum, frame):
    global global_end
    global_end = True


def train_one_epoch(model_det, model_tran, optimizers, lr_schedulers, train_loader, target_loader, device, epoch, print_freq=10, writer=None, test_func=None, save_func=None, \
det_cfg=None, args=None, i2i_cfg=None, save_cfg=None, eval_model=None, saver=None,\
a3_controller=None):
    global global_step
    global global_start
    global global_index
    global global_end
    global sample_index_iter
    
    global global_step_gan
    global is_update_memory
    d1_detection_gradient_checked = False
    
    model_det.train()
    set_d1_frozen_detector_bn_eval(model_det)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr_model', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    #metric_logger.add_meter('lr_tran', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    metric_logger.add_meter('LAMBDA', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    if det_cfg.MODEL.MODE.TRAIN_PROCESS == 'S0':
        metric_logger.add_meter('det_loss_s', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    motum_lr = 0.92
    motum_inter = 186#det_cfg.MODEL.MODE.UPDATE_TEACHER_INTER[global_index]
    update_tp_mark = det_cfg.MODEL.MODE.UPDATE_TP_MARK
    update_tp_internal = det_cfg.MODEL.MODE.UPDATE_TP_INTERNAL
    update_tp_matrix_mark = det_cfg.MODEL.MODE.UPDATE_TP_MATRIX_MARK
    update_tp_matrix_internal = det_cfg.MODEL.MODE.UPDATE_TP_MATRIX_INTERNAL

    ###
    target_loader_iter = iter(target_loader)
    for images, img_metas, targets, sc_images, sc_trans, tr_images, tr_trans, g_masks in metric_logger.log_every(train_loader, print_freq, header):
        if global_end:
             break
        global_step += 1
        global_start += 1
        start_time = time.time()
        
        _max_iters = getattr(det_cfg.SOLVER, "MAX_ITERS_EARLY_STOP", 0)
        if _max_iters > 0 and global_step >= _max_iters:
            print(f"[EARLY-STOP] global_step={global_step} >= MAX_ITERS_EARLY_STOP={_max_iters}, breaking.")
            global_end = True
            break
        for optimizer in optimizers:
            optimizer.zero_grad()
         
        images = images.to(device)
        sc_images = sc_images.to(device)
        g_masks = g_masks.to(device)
        tr_images = tr_images.to(device)
        s_targets = [t.to(device) for t in targets]
        _c1_stats = None
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'C1':
            _c1_augment = SourceAdaptiveDiversification(
                det_cfg.C1_SOURCE_DIVERSIFICATION
            )
            images, _c1_stats = _c1_augment(images, img_metas, s_targets)
        
        try:
            t_images, t_img_metas, t_targets, t_sc_images, t_sc_trans, t_tr_images, t_tr_trans, t_g_masks = next(target_loader_iter)
        except StopIteration:
            target_loader_iter = iter(target_loader)
            t_images, t_img_metas, t_targets, t_sc_images, t_sc_trans, t_tr_images, t_tr_trans, t_g_masks = next(target_loader_iter)
        
        t_images = t_images.to(device)
        t_sc_images = t_sc_images.to(device)
        #t_tr_images = t_tr_images.to(device)
        t_g_masks = t_g_masks.to(device)
        
        if det_cfg.MODEL.MODE.TRAIN_PROCESS == 'S0':
            #print(global_step)
            is_update = False
            if epoch ==0 and global_step == 1:
                is_update = True
            if det_cfg.MODEL.MODE.PJOTYPE == 'BN':
                loss_dict_s0, outputs_s = model_det(tr_images, img_metas, s_targets, mode_training=0, step_iter=global_step, \
                                                    is_update=is_update, stsc_images=sc_images, stsc_trans=sc_trans)
                det_loss_s0 = sum(list(loss_dict_s0.values()))
                det_loss_s0.backward()
                metric_logger.update(det_loss_s=det_loss_s0)
                del det_loss_s0, loss_dict_s0, outputs_s
                torch.cuda.empty_cache()
            else:
                _exp_loop = os.environ.get('SNORM_EXPERIMENT', 'E0')
                if _exp_loop == 'B2':
                    loss_dict_det, outputs_det = model_det(
                        images, img_metas, s_targets,
                        mode_training=14,
                        step_iter=global_step,
                        stpd_images=tr_images,
                        is_update=False,
                    )
                    det_only = sum(
                        value for key, value in loss_dict_det.items()
                        if not key.startswith('b2_')
                    )
                    _required_detection_losses = {
                        'rpn_cls_loss_w', 'rpn_reg_loss_w',
                        'rcnn_cls_loss_w', 'rcnn_reg_loss_w',
                    }
                    _missing = sorted(
                        _required_detection_losses - set(loss_dict_det.keys())
                    )
                    if _missing:
                        raise RuntimeError(
                            'B2 source detection losses missing: %s' % _missing
                        )
                    total_loss = sum(loss_dict_det.values())
                    total_loss.backward()
                    metric_logger.update(det_loss_s=total_loss)
                    _log_interval = int(det_cfg.B2_OBJECT_AWARE.LOG_INTERVAL)
                    if (
                            dist_utils.is_main_process()
                            and global_step % max(1, _log_interval) == 0):
                        _b2 = outputs_det['b2']
                        _gate = _b2['gate'].detach()
                        _coef = _b2['coefficient'].detach()
                        _masks = _b2['masks']
                        _obj_mean, _obj_std, _obj_count = (
                            masked_gate_statistics(
                                _gate, _masks['object_mask']
                            )
                        )
                        _small_mean, _small_std, _small_count = (
                            masked_gate_statistics(
                                _gate, _masks['small_support']
                            )
                        )
                        _bg_mean, _bg_std, _bg_count = (
                            masked_gate_statistics(
                                _gate, _masks['background_mask']
                            )
                        )
                        _inner = (
                            model_det.module
                            if hasattr(model_det, 'module') else model_det
                        )
                        _gate_grad_sq = 0.0
                        for _parameter in (
                                _inner.teacher.selective_ahdr_gate.parameters()):
                            if _parameter.grad is not None:
                                _gate_grad_sq += float(
                                    _parameter.grad.detach().pow(2).sum().item()
                                )
                        print(
                            '[B2 TRAIN] step={} total={:.6f} det={:.6f}'.format(
                                global_step,
                                float(total_loss.detach().item()),
                                float(det_only.detach().item()),
                            )
                        )
                        for _name in (
                                'b2_object_floor', 'b2_small_floor',
                                'b2_background', 'b2_tv',
                                'b2_global_anchor'):
                            print('[B2 TRAIN] {}={:.8f}'.format(
                                _name,
                                float(loss_dict_det[_name].detach().item()),
                            ))
                        print(
                            '[B2 MASK] object_coverage={:.6f} '
                            'small_coverage={:.6f} num_objects={} '
                            'num_small_objects={}'.format(
                                float(_masks['object_mask'].mean().item()),
                                float(_masks['small_support'].mean().item()),
                                sum(_masks['num_objects']),
                                sum(_masks['num_small_objects']),
                            )
                        )
                        print('[B2 GATE] global mean/std/min/max =', (
                            float(_gate.mean().item()),
                            float(_gate.std().item()),
                            float(_gate.min().item()),
                            float(_gate.max().item()),
                        ))
                        print('[B2 GATE] object mean/std =', (
                            float(_obj_mean.item()), float(_obj_std.item())
                        ))
                        print('[B2 GATE] small mean/std =', (
                            float(_small_mean.item()), float(_small_std.item())
                        ))
                        print('[B2 GATE] background mean/std =', (
                            float(_bg_mean.item()), float(_bg_std.item())
                        ))
                        print('[B2 GATE] alpha*M mean/std =', (
                            float(_coef.mean().item()),
                            float(_coef.std().item()),
                        ))
                        print('[B2 GATE] grad norm =',
                              math.sqrt(_gate_grad_sq))
                        print('[B2 TRAIN] group LRs =', [
                            (group.get('name'), group['lr'])
                            for group in optimizers[0].param_groups
                        ])
                    del total_loss, det_only, loss_dict_det, outputs_det
                    torch.cuda.empty_cache()
                elif _exp_loop == 'B1':
                    loss_dict_det, outputs_det = model_det(
                        images, img_metas, s_targets,
                        mode_training=13,
                        step_iter=global_step,
                        stpd_images=tr_images,
                        is_update=False,
                    )
                    det_only = sum(
                        value for key, value in loss_dict_det.items()
                        if not key.startswith('b1_')
                    )
                    _required_detection_losses = {
                        'rpn_cls_loss_w', 'rpn_reg_loss_w',
                        'rcnn_cls_loss_w', 'rcnn_reg_loss_w',
                    }
                    _missing_detection_losses = sorted(
                        _required_detection_losses - set(loss_dict_det.keys())
                    )
                    if _missing_detection_losses:
                        raise RuntimeError(
                            'B1 source detection losses missing: %s'
                            % _missing_detection_losses
                        )
                    total_loss = sum(list(loss_dict_det.values()))
                    total_loss.backward()
                    metric_logger.update(det_loss_s=total_loss)
                    _log_interval = int(det_cfg.SELECTIVE_AHDR.LOG_INTERVAL)
                    if (
                            dist_utils.is_main_process()
                            and global_step % max(1, _log_interval) == 0):
                        _b1 = outputs_det['b1']
                        _gate = _b1['gate'].detach()
                        _coef = _b1['coefficient'].detach()
                        _inner = (
                            model_det.module if hasattr(model_det, 'module')
                            else model_det
                        )
                        _gate_grad_sq = 0.0
                        for _parameter in (
                                _inner.teacher.selective_ahdr_gate.parameters()):
                            if _parameter.grad is not None:
                                _gate_grad_sq += float(
                                    _parameter.grad.detach().pow(2).sum().item()
                                )
                        _loss_values = {
                            key: float(value.detach().item())
                            for key, value in loss_dict_det.items()
                        }
                        print('[B1 TRAIN] step={} total={:.6f} det={:.6f}'.format(
                            global_step, float(total_loss.detach().item()),
                            float(det_only.detach().item())
                        ))
                        print('[B1 TRAIN] losses =', _loss_values)
                        print('[B1 TRAIN] gate mean/std/min/max =', (
                            float(_gate.mean().item()),
                            float(_gate.std().item()),
                            float(_gate.min().item()),
                            float(_gate.max().item()),
                        ))
                        print('[B1 TRAIN] alpha*M mean/std =', (
                            float(_coef.mean().item()),
                            float(_coef.std().item()),
                        ))
                        print('[B1 TRAIN] gate grad norm =',
                              math.sqrt(_gate_grad_sq))
                        print('[B1 TRAIN] group LRs =', [
                            (group.get('name'), group['lr'])
                            for group in optimizers[0].param_groups
                        ])
                    del total_loss, det_only, loss_dict_det, outputs_det
                    torch.cuda.empty_cache()
                elif _exp_loop == 'C3V2':
                    loss_dict_det, outputs_det = model_det(
                        tr_images, img_metas, s_targets,
                        mode_training=16, step_iter=global_step,
                        is_update=False,
                    )
                    required_detection_losses = {
                        'rpn_cls_loss_w', 'rpn_reg_loss_w',
                        'rcnn_cls_loss_w', 'rcnn_reg_loss_w',
                    }
                    missing = sorted(
                        required_detection_losses - set(loss_dict_det.keys())
                    )
                    if missing:
                        raise RuntimeError(
                            'C3-V2 source detection losses missing: %s' % missing
                        )
                    selected_detection_loss = sum(
                        value for key, value in loss_dict_det.items()
                        if key != 'c3v2_benefit_predictor'
                    )
                    total_loss = sum(loss_dict_det.values())
                    total_loss.backward()
                    metric_logger.update(det_loss_s=total_loss)
                    if (dist_utils.is_main_process()
                            and global_step % max(
                                1, int(
                                    det_cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.LOG_INTERVAL
                                )
                            ) == 0):
                        c3v2 = outputs_det['c3v2']
                        alpha = c3v2['alpha'].detach().view(-1)
                        logits = c3v2['logits'].detach().view(-1)
                        benefit = c3v2['benefit'].detach().view(-1)
                        positive = c3v2['benefit_target'].detach().view(-1)
                        positive_mask = positive > 0.5
                        negative_mask = ~positive_mask
                        alpha_pos = (
                            float(alpha[positive_mask].mean().item())
                            if bool(positive_mask.any().item()) else float('nan')
                        )
                        alpha_neg = (
                            float(alpha[negative_mask].mean().item())
                            if bool(negative_mask.any().item()) else float('nan')
                        )
                        alpha_gap = (
                            alpha_pos - alpha_neg
                            if alpha_pos == alpha_pos and alpha_neg == alpha_neg
                            else float('nan')
                        )
                        print(
                            '[C3V2 TRAIN] step={} total={:.6f} '
                            'selected_det={:.6f}'.format(
                                global_step, float(total_loss.detach().item()),
                                float(selected_detection_loss.detach().item()),
                            )
                        )
                        print('[C3V2 ALPHA] mean/std/min/max =', (
                            float(alpha.mean().item()),
                            float(alpha.std(unbiased=False).item()),
                            float(alpha.min().item()),
                            float(alpha.max().item()),
                        ))
                        print('[C3V2 ALPHA] pos/neg/gap logit_std =', (
                            alpha_pos, alpha_neg, alpha_gap,
                            float(logits.std(unbiased=False).item()),
                        ))
                        print('[C3V2 BENEFIT] mean/std/positive/negative =', (
                            float(benefit.mean().item()),
                            float(benefit.std(unbiased=False).item()),
                            float(positive.mean().item()),
                            float((1.0 - positive).mean().item()),
                        ))
                        print('[C3V2 BENEFIT] predictor_loss/accuracy =', (
                            float(c3v2['predictor_loss'].item()),
                            float(c3v2['predictor_accuracy'].item()),
                        ))
                        print('[C3V2 DETECTION] base/ahdr =', (
                            float(c3v2['base_detection_loss'].mean().item()),
                            float(c3v2['ahdr_detection_loss'].mean().item()),
                        ))
                    del total_loss, selected_detection_loss
                    del loss_dict_det, outputs_det
                    torch.cuda.empty_cache()
                elif _exp_loop == 'C3':
                    loss_dict_det, outputs_det = model_det(
                        tr_images, img_metas, s_targets,
                        mode_training=15, step_iter=global_step,
                        is_update=False,
                    )
                    required_detection_losses = {
                        'rpn_cls_loss_w', 'rpn_reg_loss_w',
                        'rcnn_cls_loss_w', 'rcnn_reg_loss_w',
                    }
                    missing = sorted(
                        required_detection_losses - set(loss_dict_det.keys())
                    )
                    if missing:
                        raise RuntimeError(
                            'C3 source detection losses missing: %s' % missing
                        )
                    selected_detection_loss = sum(
                        value for key, value in loss_dict_det.items()
                        if key != 'c3_benefit_predictor'
                    )
                    total_loss = sum(loss_dict_det.values())
                    total_loss.backward()
                    metric_logger.update(det_loss_s=total_loss)
                    if (dist_utils.is_main_process()
                            and global_step % max(
                                1, int(det_cfg.C3_DETECTION_BENEFIT.LOG_INTERVAL)
                            ) == 0):
                        c3 = outputs_det['c3']
                        alpha = c3['alpha'].detach().view(-1)
                        benefit = c3['benefit'].detach().view(-1)
                        positive = c3['benefit_target'].detach().view(-1)
                        print(
                            '[C3 TRAIN] step={} total={:.6f} '
                            'selected_det={:.6f}'.format(
                                global_step, float(total_loss.detach().item()),
                                float(selected_detection_loss.detach().item()),
                            )
                        )
                        print('[C3 ALPHA] mean/std/min/max =', (
                            float(alpha.mean().item()),
                            float(alpha.std(unbiased=False).item()),
                            float(alpha.min().item()),
                            float(alpha.max().item()),
                        ))
                        print('[C3 BENEFIT] mean/std/positive/negative =', (
                            float(benefit.mean().item()),
                            float(benefit.std(unbiased=False).item()),
                            float(positive.mean().item()),
                            float((1.0 - positive).mean().item()),
                        ))
                        print('[C3 BENEFIT] predictor_loss/accuracy =', (
                            float(c3['predictor_loss'].item()),
                            float(c3['predictor_accuracy'].item()),
                        ))
                        print('[C3 DETECTION] base/ahdr =', (
                            float(c3['base_detection_loss'].mean().item()),
                            float(c3['ahdr_detection_loss'].mean().item()),
                        ))
                    del total_loss, selected_detection_loss
                    del loss_dict_det, outputs_det
                    torch.cuda.empty_cache()
                elif _exp_loop == 'D1':
                    # Preserve the legacy AHDR reconstruction/attention losses
                    # exactly, then add a separate frozen-detector guidance
                    # forward.  mode_training=1 reaches ModuleSet mode=5,
                    # whose features_prev.detach() protects the early backbone
                    # while keeping the render_net -> detector autograd path.
                    loss_dict_s3, _ = model_det(
                        images, img_metas, s_targets, mode_training=10,
                        outputs_teacher=None, moe_mark=0,
                        stpd_images=tr_images, stpd_trans=tr_trans,
                        step_iter=global_step,
                    )
                    det_loss_s3 = 100 * sum(list(loss_dict_s3.values()))
                    det_loss_s3.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model_det.parameters(), max_norm=1.0
                    )

                    loss_dict_s5, _ = model_det(
                        images, img_metas, s_targets, mode_training=11,
                        outputs_teacher=None, moe_mark=2,
                        stpd_images=tr_images, stpd_trans=tr_trans,
                        step_iter=global_step,
                    )
                    det_loss_s5 = 100 * sum(list(loss_dict_s5.values()))
                    det_loss_s5.backward()

                    loss_dict_d1, _ = model_det(
                        tr_images, img_metas, s_targets, mode_training=1,
                        step_iter=global_step, is_update=False,
                    )
                    _d1_required_losses = (
                        'rpn_cls_loss_w', 'rpn_reg_loss_w',
                        'rcnn_cls_loss_w', 'rcnn_reg_loss_w',
                    )
                    _d1_missing_losses = [
                        key for key in _d1_required_losses
                        if key not in loss_dict_d1
                    ]
                    if _d1_missing_losses:
                        raise RuntimeError(
                            '[D1] detection losses missing: %s'
                            % _d1_missing_losses
                        )
                    d1_rpn_loss = (
                        loss_dict_d1['rpn_cls_loss_w']
                        + loss_dict_d1['rpn_reg_loss_w']
                    )
                    d1_roi_loss = (
                        loss_dict_d1['rcnn_cls_loss_w']
                        + loss_dict_d1['rcnn_reg_loss_w']
                    )
                    d1_det_loss = d1_rpn_loss + d1_roi_loss
                    _d1_det_weight = float(os.environ.get(
                        'SNORM_D1_DET_WEIGHT', '0.01'
                    ))
                    if _d1_det_weight <= 0.0:
                        raise RuntimeError(
                            '[D1] SNORM_D1_DET_WEIGHT must be positive'
                        )
                    weighted_d1_det_loss = _d1_det_weight * d1_det_loss

                    _d1_should_check_grad = (
                        not d1_detection_gradient_checked
                        or global_step % max(1, print_freq) == 0
                    )
                    _d1_detection_grad_norm = None
                    if _d1_should_check_grad:
                        _d1_inner = (
                            model_det.module
                            if hasattr(model_det, 'module') else model_det
                        )
                        _d1_render_parameters = [
                            parameter for parameter in
                            _d1_inner.teacher.proj_norm.render_net.parameters()
                            if parameter.requires_grad
                        ]
                        _d1_grads_before = [
                            (parameter.grad.detach().clone()
                             if parameter.grad is not None else None)
                            for parameter in _d1_render_parameters
                        ]

                    # This is the real DDP-compatible backward.  The optional
                    # snapshots above measure only the gradient increment from
                    # detection guidance without invoking autograd.grad(),
                    # which would bypass DDP's normal reducer contract.
                    weighted_d1_det_loss.backward()
                    if _d1_should_check_grad:
                        _d1_grad_sq = 0.0
                        _d1_grad_present = False
                        for _d1_parameter, _d1_before in zip(
                                _d1_render_parameters, _d1_grads_before):
                            _d1_after = _d1_parameter.grad
                            if _d1_after is not None:
                                _d1_increment = _d1_after.detach()
                                if _d1_before is not None:
                                    _d1_increment = (
                                        _d1_increment - _d1_before
                                    )
                                _d1_grad_present = True
                                _d1_grad_sq += float(
                                    _d1_increment.pow(2).sum().item()
                                )
                        _d1_detection_grad_norm = math.sqrt(_d1_grad_sq)
                        if (not _d1_grad_present
                                or _d1_detection_grad_norm <= 0.0):
                            raise RuntimeError(
                                '[D1] detection guidance produced no '
                                'render_net gradient'
                            )
                        d1_detection_gradient_checked = True
                    raw_ahdr_loss = det_loss_s3 + det_loss_s5
                    total_d1_loss = raw_ahdr_loss + weighted_d1_det_loss
                    metric_logger.update(det_loss_s=total_d1_loss)
                    metric_logger.update(det_loss_s3=det_loss_s3)
                    if (dist_utils.is_main_process()
                            and _d1_should_check_grad):
                        print(
                            '[D1] step={} raw_ahdr_loss={:.6f} '
                            'raw_rpn_loss={:.6f} raw_roi_loss={:.6f} '
                            'raw_det_loss={:.6f} det_weight={:.6f} '
                            'weighted_det_loss={:.6f} '
                            'render_grad_norm={:.8f}'.format(
                                global_step,
                                float(raw_ahdr_loss.detach().item()),
                                float(d1_rpn_loss.detach().item()),
                                float(d1_roi_loss.detach().item()),
                                float(d1_det_loss.detach().item()),
                                _d1_det_weight,
                                float(weighted_d1_det_loss.detach().item()),
                                _d1_detection_grad_norm,
                            )
                        )
                    del raw_ahdr_loss, total_d1_loss
                    del weighted_d1_det_loss, d1_det_loss
                    del d1_rpn_loss, d1_roi_loss, loss_dict_d1
                    del det_loss_s3, det_loss_s5
                    del loss_dict_s3, loss_dict_s5
                    torch.cuda.empty_cache()
                elif _exp_loop in ('E1', 'E2', 'A3', 'A4', 'C1', 'C2'):
                    _detector_images = images if _exp_loop == 'C1' else tr_images
                    loss_dict_det, _ = model_det(_detector_images, img_metas, s_targets, mode_training=0, step_iter=global_step, \
                                                is_update=is_update, stsc_images=sc_images, stsc_trans=sc_trans)
                    det_loss_det = sum(list(loss_dict_det.values()))
                    det_loss_det.backward()
                    metric_logger.update(det_loss_s=det_loss_det)
                    if (_exp_loop == 'C2' and dist_utils.is_main_process()
                            and global_step % max(
                                1, int(det_cfg.C2_STRUCTURE_GUIDANCE.LOG_INTERVAL)
                            ) == 0):
                        _inner = (model_det.module if hasattr(model_det, 'module')
                                  else model_det)
                        _stats = _inner.teacher.proj_norm._c2_last_stats
                        if _stats is None:
                            raise RuntimeError('C2 structure statistics missing')
                        print('[C2 TRAIN] step={} loss={:.6f} cue={} stats={}'.format(
                            global_step, float(det_loss_det.detach().item()),
                            det_cfg.C2_STRUCTURE_GUIDANCE.CUE_TYPE,
                            dict((key, float(value.item()))
                                 for key, value in _stats.items()),
                        ))
                    if (_exp_loop == 'C1' and dist_utils.is_main_process()
                            and global_step % max(1, int(det_cfg.C1_SOURCE_DIVERSIFICATION.LOG_INTERVAL)) == 0):
                        if _c1_stats.get('applied', False):
                            _print_stats = dict(
                                (key, value) for key, value in _c1_stats.items()
                                if key not in ('object_mask', 'small_mask')
                            )
                            print('[C1 AUG] step={} {}'.format(global_step, _print_stats))
                        else:
                            print('[C1 AUG] step={} applied=False'.format(global_step))
                    del det_loss_det, loss_dict_det
                    torch.cuda.empty_cache()
                else:
                    '''
                    loss_dict_s0, outputs_s = model_det(tr_images, img_metas, s_targets, mode_training=7, step_iter=global_step, \
                                                        is_update=is_update, stsc_images=sc_images, stsc_trans=sc_trans)
                    det_loss_s0 = sum(list(loss_dict_s0.values()))
                    det_loss_s0.backward()
                    metric_logger.update(det_loss_s=det_loss_s0)
                    del det_loss_s0, loss_dict_s0, outputs_s
                    torch.cuda.empty_cache()
                    '''

                    ###image render###

                    loss_dict_s3, _ = model_det(images, img_metas, s_targets, mode_training=10, outputs_teacher=None, \
                                                moe_mark=0, stpd_images=tr_images, stpd_trans=tr_trans, step_iter=global_step)

                    det_loss_s3 = 100 * sum(list(loss_dict_s3.values()))
                    det_loss_s3.backward()

                    ##################
                    torch.nn.utils.clip_grad_norm_(model_det.parameters(), max_norm=1.0)

                    loss_dict_s5, _ = model_det(images, img_metas, s_targets, mode_training=11, outputs_teacher=None, \
                                                moe_mark=2, stpd_images=tr_images, stpd_trans=tr_trans, step_iter=global_step)

                    det_loss_s5 = 100 * sum(list(loss_dict_s5.values()))
                    det_loss_s5.backward()

                    metric_logger.update(det_loss_s=det_loss_s3)
                    metric_logger.update(det_loss_s3=det_loss_s3)
                    loss_attn_val = loss_dict_s3.get('loss_attn', torch.tensor(0.0))
                    metric_logger.update(det_loss_attn=loss_attn_val)
                    
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

            _a3_snapshots = None
            if a3_controller is not None:
                if len(optimizers) != 1:
                    raise RuntimeError('A3 requires exactly one optimizer')
                _a3_snapshots = a3_controller.before_step()
            for optimizer in optimizers:
                optimizer.step()
            if a3_controller is not None:
                a3_controller.after_step(optimizers[0], _a3_snapshots)
        
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
        
        elif det_cfg.MODEL.MODE.TRAIN_PROCESS == 'S1':
            ##train detetction module
            #target domain
            loss_weight0 = 0.1
            loss_dict_t0, outputs_t0 = model_det(t_images, t_img_metas, None, domain='t', step_iter=global_step, stsc_images=t_sc_images, stsc_trans=t_sc_trans, vis=True)
            if outputs_t0 is not None:
                det_loss_t0 = loss_weight0 * sum(list(loss_dict_t0.values()))
                det_loss_t0.backward()
                metric_logger.update(loss_dict_t0=det_loss_t0)
                t_targets = copy.deepcopy(outputs_t0['dets'])
                del loss_dict_t0
                del det_loss_t0
                del outputs_t0
            torch.cuda.empty_cache()

            #source domain
            loss_weight0 = 1.0
            loss_dict_s0, outputs_s0 = model_det(images_sa1, img_metas_, ns_targets, domain='s', step_iter=global_step)
            if loss_dict_s0 is not None:
                det_loss_s0 = loss_weight0 * sum(list(loss_dict_s0.values()))
                det_loss_s0.backward()
                metric_logger.update(loss_dict_s0=det_loss_s0)
                del loss_dict_s0
                del det_loss_s0
                del outputs_s0
            torch.cuda.empty_cache()
            
            
            for optimizer in optimizers:
                    optimizer.step()

            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()

            global_step_gan += 1
            for idx in range(6):
                torch.cuda.empty_cache()
        
        LAMBDA = cosine_scheduler(det_cfg.ADV.LAMBDA_FROM, det_cfg.ADV.LAMBDA_TO, global_step)
        metric_logger.update(lr_model=optimizers[0].param_groups[0]["lr"])
        metric_logger.update(LAMBDA=LAMBDA)
        
        if global_step % print_freq == 0:
            if writer:
                writer.add_scalar('lr_model', optimizers[0].param_groups[0]['lr'], global_step=global_step)
                writer.add_scalar('LAMBDA', LAMBDA, global_step=global_step)

        _protected_ddp_exp = os.environ.get(
            'SNORM_EXPERIMENT', 'E0'
        )
        _is_b1_ddp = (
            _protected_ddp_exp in ('B1', 'B2', 'C1', 'C2', 'C3', 'C3V2')
            and dist_utils.get_world_size() > 1
        )
        _is_d1 = _protected_ddp_exp == 'D1'
        if _is_d1:
            _should_evaluate = global_step % 500 == 0
        elif _is_b1_ddp:
            _eval_env_name = (
                'C3V2_DDP_EVAL_STEPS'
                if _protected_ddp_exp == 'C3V2'
                else 'C3_DDP_EVAL_STEPS'
                if _protected_ddp_exp == 'C3'
                else 'C2_DDP_EVAL_STEPS'
                if _protected_ddp_exp == 'C2'
                else 'C1_DDP_EVAL_STEPS'
                if _protected_ddp_exp == 'C1'
                else 'B2_DDP_EVAL_STEPS'
                if _protected_ddp_exp == 'B2'
                else 'B1_DDP_EVAL_STEPS'
            )
            _eval_steps_text = os.environ.get(
                _eval_env_name, '500,1000,1500,1650,2000'
            )
            _eval_steps = set(
                int(value.strip())
                for value in _eval_steps_text.split(',')
                if value.strip()
            )
            _should_evaluate = global_step in _eval_steps
        else:
            _eval_interval = (
                1000 // max(1, (dist_utils.get_world_size() // 2))
            )
            _should_evaluate = global_step % _eval_interval == 0
        if _should_evaluate and test_func is not None:
            if _is_b1_ddp:
                _step_checkpoint_name = (
                    'model_step_latest.pth'
                    if _protected_ddp_exp == 'C3V2'
                    else 'model_step_{:06d}.pth'.format(global_step)
                )
                save_func(
                    _step_checkpoint_name,
                    'pre-evaluation checkpoint at global step {}'.format(
                        global_step
                    ),
                )
                torch.distributed.barrier()
            updated = test_func()
            if updated:
                if _is_d1:
                    save_func(
                        'best.pth',
                        'best step: {}\nFog mAP: {:.4f}\nNight mAP: {:.4f}\nSum: {:.4f}\n'.format(
                            best_step, current_foggy_mAP,
                            current_night_mAP, best_sum_mAP,
                        ),
                    )
                else:
                    save_func('best.pth', 'sum_mAP: {:.4f} (foggy: {:.4f}, night: {:.4f})'.format(best_mAP, current_foggy_mAP, current_night_mAP))
            if dist_utils.is_main_process():
                print(
                    '[{} EVAL] step={} fog={:.4f} night={:.4f} sum={:.4f}'.format(
                        _protected_ddp_exp,
                        global_step, current_foggy_mAP,
                        current_night_mAP,
                        current_foggy_mAP + current_night_mAP,
                    )
                )
                print('Best sum mAP: {:.4f} (foggy: {:.4f}, night: {:.4f})'.format(best_mAP, current_foggy_mAP, current_night_mAP))

    if det_cfg.MODEL.MODE.TRAIN_PROCESS == 'S1' and det_cfg.MODEL.MODE.UPDATE_TEACHER_MARK and epoch >= 0:
        if hasattr(model_det, 'module'):
            model_det.module.update_teacher(NET_MOMENTUM=motum_lr)
        else:
            model_det.update_teacher(NET_MOMENTUM=motum_lr)
    
    model_inner = model_det.module if hasattr(model_det, 'module') else model_det
    snorm_modules = []
    for name, m in model_inner.named_modules():
        if type(m).__name__ == 'SNorm':
            snorm_modules.append((name, m))
    _a3_active = os.environ.get('SNORM_EXPERIMENT', 'E0') in ('A3', 'A4', 'B1', 'B2')
    for name, m in snorm_modules:
        if _a3_active:
            continue
        if hasattr(m, '_clip_internal_params'):
            try: m._clip_internal_params()
            except Exception as e: pass
        if hasattr(m, 'check_nan') and hasattr(m, 'reset_ema'):
            try:
                if m.check_nan():
                    print(f"[Epoch {epoch}] WARNING: SNorm '{name}' NaN/Inf detected, resetting...")
                    m.reset_ema()
            except Exception as e: pass
        print(f"[Epoch {epoch}] SNorm '{name}': alpha_ba={m.alpha_ba.data.cpu().numpy()}, alpha_bu={m.alpha_bu.data.cpu().numpy()}")

    _pjotype = det_cfg.MODEL.MODE.PJOTYPE
    if _pjotype != 'BN' and not _a3_active:
        if hasattr(model_det, 'module'):
            model_det.module.teacher.proj_norm.compute_norm_parameters(device=device)
        else:
            model_det.teacher.proj_norm.compute_norm_parameters(device=device)
    elif _a3_active:
        print('[A3/A4/B1/B2] proj_norm frozen: skipped epoch state clipping/reset/compute_norm_parameters')

def main(det_cfg, args):
    global best_mAP, best_sum_mAP, best_step
    global current_foggy_mAP, current_night_mAP
    print('----num_workers:', det_cfg.DATALOADER.NUM_WORKERS)
    
    # 只在非测试模式下加载训练数据
    if not args.test_only:
        train_loader = build_data_loaders(det_cfg.DATASETS.TRAINS, transforms=det_cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                      batch_size=det_cfg.SOLVER.BATCH_SIZE, num_workers=det_cfg.DATALOADER.NUM_WORKERS, shuffle=False)
        target_loader = build_data_loaders(det_cfg.DATASETS.TARGETS, transforms=det_cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                       batch_size=det_cfg.SOLVER.BATCH_SIZE, num_workers=0, shuffle=True)
        if os.environ.get('SNORM_EXPERIMENT', 'E0') in ('B1', 'B2', 'C1', 'C2', 'C3', 'C3V2'):
            _configured_batch = int(det_cfg.SOLVER.BATCH_SIZE)
            _loader_batch = int(train_loader.batch_sampler.batch_size)
            if _loader_batch != _configured_batch:
                raise RuntimeError(
                    '%s DataLoader batch mismatch: cfg=%s loader=%s'
                    % (os.environ.get('SNORM_EXPERIMENT', 'E0'), _configured_batch, _loader_batch)
                )
            if args.distributed and _configured_batch != 2:
                raise RuntimeError(
                    '%s DDP requires per-rank SOLVER.BATCH_SIZE=2; got %s'
                    % (os.environ.get('SNORM_EXPERIMENT', 'E0'), _configured_batch)
                )
            if dist_utils.is_main_process():
                _world_size = dist_utils.get_world_size()
                print('[{} DDP BATCH] cfg per-rank ='.format(os.environ.get('SNORM_EXPERIMENT', 'E0')), _configured_batch)
                print('[{} DDP BATCH] loader per-rank ='.format(os.environ.get('SNORM_EXPERIMENT', 'E0')), _loader_batch)
                print('[{} DDP BATCH] global ='.format(os.environ.get('SNORM_EXPERIMENT', 'E0')),
                      _loader_batch * _world_size)
                print('[{} DDP BATCH] dataset ='.format(os.environ.get('SNORM_EXPERIMENT', 'E0')),
                      len(train_loader.dataset))
                print('[{} DDP BATCH] sampler samples/rank ='.format(os.environ.get('SNORM_EXPERIMENT', 'E0')),
                      len(train_loader.batch_sampler.sampler))
                print('[{} DDP BATCH] optimizer steps/epoch ='.format(os.environ.get('SNORM_EXPERIMENT', 'E0')),
                      len(train_loader))
    
    # 加载测试数据
    test_loaders = build_data_loaders(det_cfg.DATASETS.TESTS, transforms=det_cfg.INPUT.TRANSFORMS_TEST, is_train=False,
                                      distributed=args.distributed, num_workers=0)
    
    ###load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_det = build_detectors(det_cfg)
    model_det.to(device)

    # D1 has a stricter whitelist than every detector-training experiment.
    # Apply it before DDP constructs reducer hooks; optimizer setup below
    # repeats and audits the same policy.
    if (args.distributed
            and os.environ.get('SNORM_EXPERIMENT', 'E0') == 'D1'):
        for _pre_d1_parameter in model_det.teacher.parameters():
            _pre_d1_parameter.requires_grad_(False)
        for _pre_d1_parameter in model_det.dis.parameters():
            _pre_d1_parameter.requires_grad_(False)
        for _pre_d1_parameter in (
                model_det.teacher.proj_norm.render_net.parameters()):
            _pre_d1_parameter.requires_grad_(True)
        _pre_d1_unexpected = [
            name for name, parameter in model_det.teacher.named_parameters()
            if (parameter.requires_grad
                and not name.startswith('proj_norm.render_net.'))
        ]
        if _pre_d1_unexpected:
            raise RuntimeError(
                '[D1 PRE-DDP] unexpected trainable parameters: %s'
                % _pre_d1_unexpected
            )
        print('[D1 PRE-DDP] render_net-only requires_grad policy applied')

    # DDP must observe the final B1 requires_grad whitelist when reducer hooks
    # are constructed. The later optimizer setup repeats this idempotently.
    if (
            args.distributed
            and os.environ.get('SNORM_EXPERIMENT', 'E0') in ('B1', 'B2', 'C1', 'C2', 'C3', 'C3V2')):
        _pre_ddp_exp = os.environ.get('SNORM_EXPERIMENT', 'E0')
        if _pre_ddp_exp == 'B2':
            configure_b2_trainable_parameters(model_det)
        elif _pre_ddp_exp in ('C1', 'C2', 'C3', 'C3V2'):
            _pre_ddp_teacher = model_det.teacher
            if _pre_ddp_exp in ('C3', 'C3V2'):
                for p in _pre_ddp_teacher.parameters():
                    p.requires_grad_(False)
            for p in _pre_ddp_teacher.backbone_0.parameters(): p.requires_grad_(False)
            for p in _pre_ddp_teacher.backbone_1.parameters(): p.requires_grad_(False)
            for p in _pre_ddp_teacher.backbone_2.parameters(): p.requires_grad_(False)
            for p in _pre_ddp_teacher.backbone_3.parameters(): p.requires_grad_(True)
            for p in _pre_ddp_teacher.proj_norm.parameters(): p.requires_grad_(False)
            for p in _pre_ddp_teacher.rpn_norm.parameters(): p.requires_grad_(True)
            for p in _pre_ddp_teacher.box_head_norm.parameters(): p.requires_grad_(True)
            if _pre_ddp_exp == 'C3':
                for p in _pre_ddp_teacher.detection_benefit_predictor.parameters():
                    p.requires_grad_(True)
            elif _pre_ddp_exp == 'C3V2':
                for p in _pre_ddp_teacher.detection_semantic_benefit_predictor.parameters():
                    p.requires_grad_(True)
            for p in model_det.dis.parameters(): p.requires_grad_(False)
            print('[{} PRE-DDP] applied E2 detector requires_grad policy'.format(
                _pre_ddp_exp
            ))
        else:
            configure_b1_trainable_parameters(model_det)
    
    model_without_ddp = model_det
    model_tran_without_ddp = None
    if args.distributed:
        print('--set distributed')
        model_det = DistributedDataParallel(
            convert_sync_batchnorm(model_det),
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
       
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
        model_without_ddp = model_det.module
    
    current_epoch = -1
    _default_pretrain = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pth')
    _load_path = args.resume if args.resume else (_default_pretrain if os.path.exists(_default_pretrain) else '')
    _anchor_sizes = tuple(int(v) for v in det_cfg.MODEL.RPN.ANCHOR_SIZES)
    _aspect_ratios = tuple(float(v) for v in det_cfg.MODEL.RPN.ASPECT_RATIOS)
    _is_small_anchor_a = _anchor_sizes == (32, 64, 128, 256, 512)
    try:
        _git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
    except Exception:
        _git_sha = 'unavailable'
    _fixed_e2_checkpoint = (
        '/home/ubuntu/hdx/snorm_E2_repro/debug/city_2_foggy/'
        's0_dbias_color_E2_zdh_parallel/E2/best.pth'
    )
    if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'D1':
        if os.path.abspath(_load_path or '') != _fixed_e2_checkpoint:
            raise RuntimeError(
                'D1 must initialize from read-only E2 checkpoint: %s'
                % _fixed_e2_checkpoint
            )
    if os.environ.get('SNORM_EXPERIMENT', 'E0') in ('B1', 'B2', 'C1', 'C2', 'C3', 'C3V2', 'D1'):
        _protected_exp = os.environ.get('SNORM_EXPERIMENT', 'E0')
        if os.path.abspath(_load_path or '') != _fixed_e2_checkpoint:
            raise RuntimeError(
                '%s must initialize from fixed E2 checkpoint: %s'
                % (_protected_exp, _fixed_e2_checkpoint)
            )
        if os.environ.get('SNORM_ABLATION', 'full') != 'full':
            raise RuntimeError(
                '%s requires full six-step AHDR' % _protected_exp
            )
        if int(os.environ.get('SNORM_STEPS', '6')) != 6:
            raise RuntimeError('%s requires SNORM_STEPS=6' % _protected_exp)
        if abs(float(os.environ.get('SNORM_RES_W', '0.6')) - 0.6) > 1e-12:
            raise RuntimeError(
                '%s requires SNORM_RES_W=0.6' % _protected_exp
            )
        if list(det_cfg.DATASETS.TRAINS) != ['bdd_daytime_clear_train']:
            raise RuntimeError(
                '%s training must remain clear-source-only' % _protected_exp
            )
        if list(det_cfg.DATASETS.TESTS) != [
                'bdd_daytime_fog_test', 'bdd_night_sun_test']:
            raise RuntimeError(
                '%s Fog/Night evaluation protocol changed' % _protected_exp
            )
        if _protected_exp == 'D1':
            if _anchor_sizes != (64, 128, 256, 512):
                raise RuntimeError(
                    'D1 must use original 12-anchor configuration '
                    '(64, 128, 256, 512), got %s'
                    % (str(_anchor_sizes),)
                )
        if _protected_exp == 'C1':
            if _anchor_sizes != (64, 128, 256, 512):
                raise RuntimeError(
                    'C1 must use original 12-anchor configuration '
                    '(64, 128, 256, 512), got %s'
                    % (str(_anchor_sizes),)
                )
            if not bool(det_cfg.C1_SOURCE_DIVERSIFICATION.ENABLED):
                raise RuntimeError(
                    'C1_SOURCE_DIVERSIFICATION.ENABLED must be True'
                )
        if _protected_exp == 'C2':
            if _anchor_sizes != (64, 128, 256, 512):
                raise RuntimeError(
                    'C2 must use original 12-anchor configuration '
                    '(64, 128, 256, 512), got %s' % (str(_anchor_sizes),)
                )
            if not bool(det_cfg.C2_STRUCTURE_GUIDANCE.ENABLED):
                raise RuntimeError('C2_STRUCTURE_GUIDANCE.ENABLED must be True')
            if bool(det_cfg.C1_SOURCE_DIVERSIFICATION.ENABLED):
                raise RuntimeError('C2 must not enable C1 source diversification')
            if bool(det_cfg.SELECTIVE_AHDR.ENABLED):
                raise RuntimeError('C2 must not enable B1/B2 Selective-AHDR')
        if _protected_exp == 'C3':
            if _anchor_sizes != (64, 128, 256, 512):
                raise RuntimeError(
                    'C3 must use original 12-anchor configuration '
                    '(64, 128, 256, 512), got %s' % (str(_anchor_sizes),)
                )
            if not bool(det_cfg.C3_DETECTION_BENEFIT.ENABLED):
                raise RuntimeError('C3_DETECTION_BENEFIT.ENABLED must be True')
            if bool(det_cfg.C1_SOURCE_DIVERSIFICATION.ENABLED):
                raise RuntimeError('C3 must not enable C1 source diversification')
            if bool(det_cfg.C2_STRUCTURE_GUIDANCE.ENABLED):
                raise RuntimeError('C3 must not enable C2 structure guidance')
            if bool(det_cfg.SELECTIVE_AHDR.ENABLED):
                raise RuntimeError('C3 must not enable B1/B2 Selective-AHDR')
            if bool(det_cfg.B2_OBJECT_AWARE.ENABLED):
                raise RuntimeError('C3 must not enable B2 object-aware losses')
        if _protected_exp == 'C3V2':
            if _anchor_sizes != (64, 128, 256, 512):
                raise RuntimeError(
                    'C3-V2 must use original 12-anchor configuration '
                    '(64, 128, 256, 512), got %s' % (str(_anchor_sizes),)
                )
            if not bool(
                    det_cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.ENABLED):
                raise RuntimeError(
                    'C3V2_DETECTION_SEMANTIC_BENEFIT.ENABLED must be True'
                )
            if bool(det_cfg.C3_DETECTION_BENEFIT.ENABLED):
                raise RuntimeError('C3-V2 must not enable C3-V1 predictor')
            if bool(det_cfg.C1_SOURCE_DIVERSIFICATION.ENABLED):
                raise RuntimeError('C3-V2 must not enable C1')
            if bool(det_cfg.C2_STRUCTURE_GUIDANCE.ENABLED):
                raise RuntimeError('C3-V2 must not enable C2')
            if bool(det_cfg.SELECTIVE_AHDR.ENABLED):
                raise RuntimeError('C3-V2 must not enable B1/B2 Selective-AHDR')
            if bool(det_cfg.B2_OBJECT_AWARE.ENABLED):
                raise RuntimeError('C3-V2 must not enable B2 object-aware losses')
        if (
                _protected_exp == 'B2'
                and not bool(det_cfg.B2_OBJECT_AWARE.ENABLED)):
            raise RuntimeError('B2_OBJECT_AWARE.ENABLED must be True')
    print('[EXPERIMENT START]')
    print('experiment =', (
        'C3V2_DETECTION_SEMANTIC_BENEFIT_AHDR'
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'C3V2'
        else 'C3_DETECTION_BENEFIT_AWARE_AHDR'
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'C3'
        else 'C2_STRUCTURE_GUIDED_AHDR'
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'C2'
        else 'C1_SOURCE_ADAPTIVE_DIVERSIFICATION'
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'C1'
        else 'B2_OBJECT_AWARE_SELECTIVE_AHDR'
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'B2'
        else 'B1_PROTECTED_SMALLANCHOR_SELECTIVE_AHDR'
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'B1'
        else 'A4_PROTECTED_RPN'
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'A4'
        else 'A3_PROTECTED_SMALLANCHOR'
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'A3'
        else 'A_SMALLANCHOR' if _is_small_anchor_a
        else os.environ.get('SNORM_EXPERIMENT', 'E0')
    ))
    print('git SHA =', _git_sha)
    print('anchor sizes =', _anchor_sizes)
    print('number of anchors =', len(_anchor_sizes) * len(_aspect_ratios))
    print('source train dataset =', list(det_cfg.DATASETS.TRAINS))
    print('target/test datasets =', list(det_cfg.DATASETS.TESTS))
    print('checkpoint =', _load_path if _load_path else '<none>')
    print('checkpoint initialization mode =', (
        'AUDITED_SMALL_ANCHOR_COMPAT' if _is_small_anchor_a and _load_path
        else 'STANDARD_EXACT_LOAD' if _load_path else 'RANDOM_INITIALIZATION'
    ))
    if _load_path:
        print('Loading checkpoint from {} ...'.format(_load_path))
        if _is_small_anchor_a:
            _load_result = load_small_anchor_checkpoint(
                model_without_ddp, _load_path,
                old_anchor_sizes=(64, 128, 256, 512),
                new_anchor_sizes=_anchor_sizes,
            )
            checkpoint = _load_result['checkpoint']
        else:
            checkpoint = torch.load(_load_path, map_location='cpu')
            _state = checkpoint['model'] if 'model' in checkpoint else checkpoint
            _incompatible = model_without_ddp.load_state_dict(
                _state, strict=False
            )
            if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'C3':
                _allowed_missing = {
                    'teacher.detection_benefit_predictor.encoder.weight',
                    'teacher.detection_benefit_predictor.encoder.bias',
                    'teacher.detection_benefit_predictor.predictor.weight',
                    'teacher.detection_benefit_predictor.predictor.bias',
                }
                _missing = set(_incompatible.missing_keys)
                _unexpected = set(_incompatible.unexpected_keys)
                if _missing != _allowed_missing or _unexpected:
                    raise RuntimeError(
                        'C3 checkpoint incompatibility: missing=%s unexpected=%s'
                        % (sorted(_missing), sorted(_unexpected))
                    )
                print('[C3 CHECKPOINT] only predictor parameters are new')
            elif os.environ.get('SNORM_EXPERIMENT', 'E0') == 'C3V2':
                _allowed_missing = {
                    'teacher.detection_semantic_benefit_predictor.projection.weight',
                    'teacher.detection_semantic_benefit_predictor.projection.bias',
                    'teacher.detection_semantic_benefit_predictor.encoder.weight',
                    'teacher.detection_semantic_benefit_predictor.encoder.bias',
                    'teacher.detection_semantic_benefit_predictor.predictor.weight',
                    'teacher.detection_semantic_benefit_predictor.predictor.bias',
                }
                _missing = set(_incompatible.missing_keys)
                _unexpected = set(_incompatible.unexpected_keys)
                if _missing != _allowed_missing or _unexpected:
                    raise RuntimeError(
                        'C3-V2 checkpoint incompatibility: '
                        'missing=%s unexpected=%s'
                        % (sorted(_missing), sorted(_unexpected))
                    )
                print('[C3-V2 CHECKPOINT] only semantic predictor parameters are new')
        print('checkpoint load completed')
        if 'current_epoch' in checkpoint and args.use_current_epoch:
            current_epoch = int(checkpoint['current_epoch'])
    else:
        print('WARNING: no pretrained checkpoint; using random initialization')
    
    work_dir = det_cfg.WORK_DIR
    _exp_wd = os.environ.get('SNORM_EXPERIMENT', 'E0')
    if _exp_wd != 'E0':
        _work_suffix = 'E2' if _exp_wd in ('A3', 'A4', 'B1', 'B2', 'C1', 'C2', 'C3', 'C3V2') else _exp_wd
        work_dir = os.path.join(work_dir, _work_suffix)
        os.makedirs(work_dir, exist_ok=True)
        print(f'  [WORK_DIR] SNORM_EXPERIMENT={_exp_wd}, work_dir={work_dir}')
    if _exp_wd == 'D1':
        _d1_best_path = os.path.join(work_dir, 'best.txt')
        best_mAP = 0.0
        best_sum_mAP = 0.0
        best_step = -1
        current_foggy_mAP = 0.0
        current_night_mAP = 0.0
        if os.path.exists(_d1_best_path):
            try:
                with open(_d1_best_path, 'r') as _d1_best_file:
                    _d1_best_values = {}
                    for _d1_line in _d1_best_file:
                        _d1_line = _d1_line.strip()
                        if not _d1_line:
                            continue
                        _d1_key, _d1_value = _d1_line.split(':', 1)
                        _d1_best_values[_d1_key.strip()] = _d1_value.strip()
                _d1_required_best_keys = (
                    'best step', 'Fog mAP', 'Night mAP', 'Sum'
                )
                _d1_missing_best_keys = [
                    key for key in _d1_required_best_keys
                    if key not in _d1_best_values
                ]
                if _d1_missing_best_keys:
                    raise ValueError(
                        'missing keys: %s' % _d1_missing_best_keys
                    )
                best_step = int(_d1_best_values['best step'])
                current_foggy_mAP = float(_d1_best_values['Fog mAP'])
                current_night_mAP = float(_d1_best_values['Night mAP'])
                best_sum_mAP = float(_d1_best_values['Sum'])
                if not all(math.isfinite(value) for value in (
                        current_foggy_mAP, current_night_mAP,
                        best_sum_mAP)):
                    raise ValueError('non-finite metric')
                best_mAP = best_sum_mAP
            except Exception as _d1_best_error:
                raise RuntimeError(
                    '[D1] failed to parse historical best file %s: %s'
                    % (_d1_best_path, _d1_best_error)
                )
            print('[D1] historical best Sum: {:.4f}'.format(
                best_sum_mAP
            ))
        else:
            print('[D1] no historical best found')
    if args.test_only:
        evaluation(model_det, test_loaders, device, types=det_cfg.TEST.EVAL_TYPES, output_dir=work_dir)
        return
    
    ##set optimizer##
    print('--set optimizer')
    model_paramters = []
    model_eval_parameters = []
    a3_controller = None
    if det_cfg.MODEL.MODE.TRAIN_PROCESS == 'S0':
          if det_cfg.MODEL.MODE.PJOTYPE == 'BN':
              model_paramters += [{'params': model_without_ddp.teacher.parameters(),'lr':det_cfg.SOLVER.LR}]
          else:
              _exp = os.environ.get('SNORM_EXPERIMENT', 'E0')
              _base_lr = det_cfg.SOLVER.LR
              _teacher = model_without_ddp.teacher
              _dis = model_without_ddp.dis
              _proj = _teacher.proj_norm
              _rn = _proj.render_net
              _sp = getattr(_proj, 'sigma_weight_pred', None)
              def _count(_params): return sum(p.numel() for p in _params)
              if _exp == 'D1':
                  _d1_incompatible = []
                  if bool(det_cfg.C1_SOURCE_DIVERSIFICATION.ENABLED):
                      _d1_incompatible.append('C1_SOURCE_DIVERSIFICATION')
                  if bool(det_cfg.SELECTIVE_AHDR.ENABLED):
                      _d1_incompatible.append('SELECTIVE_AHDR')
                  if bool(det_cfg.C2_STRUCTURE_GUIDANCE.ENABLED):
                      _d1_incompatible.append('C2_STRUCTURE_GUIDANCE')
                  if bool(det_cfg.C3_DETECTION_BENEFIT.ENABLED):
                      _d1_incompatible.append('C3_DETECTION_BENEFIT')
                  if bool(
                          det_cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.ENABLED):
                      _d1_incompatible.append(
                          'C3V2_DETECTION_SEMANTIC_BENEFIT'
                      )
                  if bool(det_cfg.B2_OBJECT_AWARE.ENABLED):
                      _d1_incompatible.append('B2_OBJECT_AWARE')
                  if _d1_incompatible:
                      raise RuntimeError(
                          '[D1] incompatible mechanisms must be disabled: %s'
                          % _d1_incompatible
                      )

                  for _parameter in _teacher.parameters():
                      _parameter.requires_grad_(False)
                  for _parameter in _dis.parameters():
                      _parameter.requires_grad_(False)
                  _d1_render_parameters = list(_rn.parameters())
                  for _parameter in _d1_render_parameters:
                      _parameter.requires_grad_(True)

                  _d1_trainable_names = [
                      name for name, parameter in
                      _teacher.named_parameters()
                      if parameter.requires_grad
                  ]
                  for _name in _d1_trainable_names:
                      if not _name.startswith('proj_norm.render_net.'):
                          raise RuntimeError(
                              '[D1] unexpected trainable teacher parameter: '
                              + _name
                          )
                  _d1_render_count = _count(
                      parameter for parameter in _d1_render_parameters
                      if parameter.requires_grad
                  )
                  if _d1_render_count == 0:
                      raise RuntimeError(
                          '[D1] render_net has no trainable parameters'
                      )

                  _d1_detector_modules = (
                      ('backbone_0', _teacher.backbone_0),
                      ('backbone_1', _teacher.backbone_1),
                      ('backbone_2', _teacher.backbone_2),
                      ('backbone_3', _teacher.backbone_3),
                      ('rpn_norm', _teacher.rpn_norm),
                      ('box_head_norm', _teacher.box_head_norm),
                      ('rpn_eval', _teacher.rpn_eval),
                      ('box_head_eval', _teacher.box_head_eval),
                  )
                  _d1_detector_count = sum(
                      parameter.numel()
                      for _, module in _d1_detector_modules
                      for parameter in module.parameters()
                      if parameter.requires_grad
                  )
                  if _d1_detector_count != 0:
                      raise RuntimeError(
                          '[D1] detector trainable parameter count must be 0; '
                          'got %s' % _d1_detector_count
                      )
                  _d1_det_weight = float(os.environ.get(
                      'SNORM_D1_DET_WEIGHT', '0.01'
                  ))
                  if _d1_det_weight <= 0.0:
                      raise RuntimeError(
                          '[D1] SNORM_D1_DET_WEIGHT must be positive'
                      )
                  model_paramters += [{
                      'params': _d1_render_parameters,
                      'lr': _base_lr,
                      'name': 'd1_render_net',
                  }]
                  print('  [D1] Frozen-Detector Task-Guided AHDR')
                  print('  [D1] render_net trainable params:',
                        _d1_render_count)
                  print('  [D1] detector trainable params:',
                        _d1_detector_count)
                  print('  [D1] optimizer params:', _d1_render_count)
                  print('  [D1] sigma_weight_pred trainable:',
                        bool(_sp is not None and any(
                            parameter.requires_grad
                            for parameter in _sp.parameters()
                        )))
                  print('  [D1] DET_WEIGHT:', _d1_det_weight)
                  a3_controller = None
              elif _exp in ('B1', 'B2'):
                  if not _is_small_anchor_a:
                      raise RuntimeError(
                          '%s requires protected anchor sizes' % _exp
                      )
                  if not bool(det_cfg.SELECTIVE_AHDR.ENABLED):
                      raise RuntimeError(
                          '%s requires SELECTIVE_AHDR.ENABLED' % _exp
                      )
                  if float(_base_lr) != float(B2_OUTPUT_LR if _exp == 'B2' else B1_OUTPUT_LR):
                      raise RuntimeError(
                          '%s requires SOLVER.LR=%s; got %s'
                          % (_exp, B2_OUTPUT_LR if _exp == 'B2' else B1_OUTPUT_LR, _base_lr)
                      )
                  _gate_lr = float(det_cfg.SELECTIVE_AHDR.GATE_LR)
                  if _exp == 'B2':
                      _outputs, _gate, _gate_names = (
                          configure_b2_trainable_parameters(
                              model_without_ddp
                          )
                      )
                      _output_group = 'b2_new_32_anchor_output'
                      _gate_group = 'b2_object_aware_gate'
                  else:
                      _outputs, _gate, _gate_names = (
                          configure_b1_trainable_parameters(
                              model_without_ddp
                          )
                      )
                      _output_group = 'b1_new_32_anchor_output'
                      _gate_group = 'b1_selective_ahdr_gate'
                  model_paramters += [{
                      'params': _outputs,
                      'lr': B2_OUTPUT_LR if _exp == 'B2' else B1_OUTPUT_LR,
                      'name': _output_group,
                  }, {
                      'params': _gate,
                      'lr': _gate_lr,
                      'name': _gate_group,
                  }]
                  a3_controller = ProtectedSmallAnchorStep(
                      model_without_ddp,
                      extra_trainable_names=_gate_names,
                  )
              elif _exp in ('C1', 'C2', 'C3', 'C3V2'):
                  print('  [{}] 12-anchor detector training policy = E2'.format(_exp))
                  if _exp in ('C3', 'C3V2'):
                      for p in _teacher.parameters():
                          p.requires_grad_(False)
                  for p in _teacher.backbone_0.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_1.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_2.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_3.parameters(): p.requires_grad_(True)
                  for p in _teacher.proj_norm.parameters(): p.requires_grad_(False)
                  for p in _teacher.rpn_norm.parameters(): p.requires_grad_(True)
                  for p in _teacher.box_head_norm.parameters(): p.requires_grad_(True)
                  for p in _dis.parameters(): p.requires_grad_(False)
                  if _exp == 'C3':
                      for p in _teacher.detection_benefit_predictor.parameters():
                          p.requires_grad_(True)
                  elif _exp == 'C3V2':
                      for p in _teacher.detection_semantic_benefit_predictor.parameters():
                          p.requires_grad_(True)

                  _b3p = _count(_teacher.backbone_3.parameters())
                  _rp = _count(_teacher.rpn_norm.parameters())
                  _bp = _count(_teacher.box_head_norm.parameters())

                  print('  [{}] backbone_3 params:'.format(_exp), _b3p,
                        'rpn_norm params:', _rp,
                        'box_head_norm params:', _bp)
                  print('  [{}] backbone_3 lr:'.format(_exp), _base_lr * 0.1,
                        'detector head lr:', _base_lr)

                  model_paramters += [{
                      'params': _teacher.backbone_3.parameters(),
                      'lr': _base_lr * 0.1,
                      'name': _exp.lower() + '_backbone_3',
                  }]
                  model_paramters += [{
                      'params': _teacher.rpn_norm.parameters(),
                      'lr': _base_lr,
                      'name': _exp.lower() + '_rpn_norm',
                  }]
                  model_paramters += [{
                      'params': _teacher.box_head_norm.parameters(),
                      'lr': _base_lr,
                      'name': _exp.lower() + '_box_head_norm',
                  }]
                  if _exp == 'C3':
                      _predictor = _teacher.detection_benefit_predictor
                      _predictor_lr = float(
                          det_cfg.C3_DETECTION_BENEFIT.PREDICTOR_LR
                      )
                      _predictor_params = _count(_predictor.parameters())
                      print('  [C3] benefit predictor params:',
                            _predictor_params, 'lr:', _predictor_lr)
                      model_paramters += [{
                          'params': _predictor.parameters(),
                          'lr': _predictor_lr,
                          'name': 'c3_detection_benefit_predictor',
                      }]
                  elif _exp == 'C3V2':
                      _predictor = (
                          _teacher.detection_semantic_benefit_predictor
                      )
                      _predictor_lr = float(
                          det_cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.PREDICTOR_LR
                      )
                      _predictor_params = _count(_predictor.parameters())
                      print('  [C3-V2] semantic predictor params:',
                            _predictor_params, 'lr:', _predictor_lr)
                      model_paramters += [{
                          'params': _predictor.parameters(),
                          'lr': _predictor_lr,
                          'name': 'c3v2_detection_semantic_predictor',
                      }]

                  a3_controller = None
              elif _exp == 'A4':
                  if not _is_small_anchor_a:
                      raise RuntimeError('A4 requires anchor sizes (32,64,128,256,512)')
                  if float(_base_lr) != float(A4_OUTPUT_LR):
                      raise RuntimeError(
                          'A4 requires SOLVER.LR=%s; got %s'
                          % (A4_OUTPUT_LR, _base_lr)
                      )
                  _a4_outputs, _a4_conv, _a4_conv_names = (
                      configure_a4_trainable_parameters(model_without_ddp)
                  )
                  model_paramters += [{
                      'params': _a4_outputs,
                      'lr': A4_OUTPUT_LR,
                      'name': 'a4_new_32_anchor_output',
                  }, {
                      'params': _a4_conv,
                      'lr': A4_SHARED_CONV_LR,
                      'name': 'a4_rpn_shared_conv',
                  }]
                  a3_controller = ProtectedSmallAnchorStep(
                      model_without_ddp,
                      extra_trainable_names=_a4_conv_names,
                  )
              elif _exp == 'A3':
                  if not _is_small_anchor_a:
                      raise RuntimeError('A3 requires anchor sizes (32,64,128,256,512)')
                  _a3_parameters = configure_a3_trainable_parameters(model_without_ddp)
                  model_paramters += [{
                      'params': _a3_parameters,
                      'lr': _base_lr,
                      'name': 'a3_new_32_anchor_output_channels',
                  }]
                  a3_controller = ProtectedSmallAnchorStep(model_without_ddp)
              elif _exp == 'E1':
                  print('  [E1] SNORM_EXPERIMENT=E1: 冻结 backbone+proj_norm；训练 rpn_norm + box_head_norm')
                  for p in _teacher.backbone_0.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_1.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_2.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_3.parameters(): p.requires_grad_(False)
                  for p in _teacher.proj_norm.parameters(): p.requires_grad_(False)
                  for p in _teacher.rpn_norm.parameters(): p.requires_grad_(True)
                  for p in _teacher.box_head_norm.parameters(): p.requires_grad_(True)
                  for p in _dis.parameters(): p.requires_grad_(False)
                  _rp = _count(_teacher.rpn_norm.parameters())
                  _bp = _count(_teacher.box_head_norm.parameters())
                  print('  [E1] rpn_norm params:', _rp, 'box_head_norm params:', _bp, 'total:', _rp + _bp)
                  model_paramters += [{'params': _teacher.rpn_norm.parameters(), 'lr': _base_lr}]
                  model_paramters += [{'params': _teacher.box_head_norm.parameters(), 'lr': _base_lr}]
              elif _exp == 'E2':
                  print('  [E2] SNORM_EXPERIMENT=E2: 冻结 backbone_0/1/2+proj_norm；训练 backbone_3(lr×0.1) + rpn_norm + box_head_norm')
                  for p in _teacher.backbone_0.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_1.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_2.parameters(): p.requires_grad_(False)
                  for p in _teacher.proj_norm.parameters(): p.requires_grad_(False)
                  for p in _teacher.backbone_3.parameters(): p.requires_grad_(True)
                  for p in _teacher.rpn_norm.parameters(): p.requires_grad_(True)
                  for p in _teacher.box_head_norm.parameters(): p.requires_grad_(True)
                  for p in _dis.parameters(): p.requires_grad_(False)
                  _b3p = _count(_teacher.backbone_3.parameters())
                  _rp = _count(_teacher.rpn_norm.parameters())
                  _bp = _count(_teacher.box_head_norm.parameters())
                  print('  [E2] backbone_3 params:', _b3p, '(lr×0.1), rpn_norm:', _rp, 'box_head_norm:', _bp, 'total:', _b3p + _rp + _bp)
                  model_paramters += [{'params': _teacher.backbone_3.parameters(), 'lr': _base_lr * 0.1}]
                  model_paramters += [{'params': _teacher.rpn_norm.parameters(), 'lr': _base_lr}]
                  model_paramters += [{'params': _teacher.box_head_norm.parameters(), 'lr': _base_lr}]
              else:
                  _exp = 'E0'
                  if _sp is not None:
                      print('  [E0] baseline: 冻结 teacher+dis；训练 render_net + sigma_weight_pred')
                      for p in _teacher.parameters(): p.requires_grad_(False)
                      for p in _dis.parameters(): p.requires_grad_(False)
                      for p in _rn.parameters(): p.requires_grad_(True)
                      for p in _sp.parameters(): p.requires_grad_(True)
                      _rn_params = _count(_rn.parameters())
                      _sg_params = _count(_sp.parameters())
                      print('  [E0] render_net params:', _rn_params, 'sigma_weight_pred params:', _sg_params)
                      model_paramters += [{'params': _rn.parameters(), 'lr': _base_lr}]
                      model_paramters += [{'params': _sp.parameters(), 'lr': _base_lr}]
                  else:
                      for p in _teacher.parameters(): p.requires_grad_(False)
                      for p in _dis.parameters(): p.requires_grad_(False)
                      for p in _rn.parameters(): p.requires_grad_(True)
                      model_paramters += [{'params': _rn.parameters(), 'lr': _base_lr}]
                      model_paramters += [{'params': _dis.parameters(), 'lr': _base_lr}]
              print('  [SNORM_EXPERIMENT] active =', _exp)
          if _is_small_anchor_a:
              print('[A_SMALLANCHOR TRAINABLE GROUPS]')
              for _group_index, _group in enumerate(model_paramters):
                  _group_params = list(_group['params'])
                  _group['params'] = _group_params
                  print('group={} lr={} trainable_params={}'.format(
                      _group_index, _group.get('lr', det_cfg.SOLVER.LR),
                      sum(p.numel() for p in _group_params if p.requires_grad)
                  ))
          optimizer_model = torch.optim.Adam(model_paramters, det_cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=det_cfg.SOLVER.WEIGHT_DECAY)
          if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'D1':
              _d1_allowed_optimizer_ids = set(
                  id(parameter)
                  for parameter in
                  model_without_ddp.teacher.proj_norm.render_net.parameters()
              )
              _d1_optimizer_ids = set(
                  id(parameter)
                  for group in optimizer_model.param_groups
                  for parameter in group['params']
              )
              if _d1_optimizer_ids != _d1_allowed_optimizer_ids:
                  raise RuntimeError(
                      '[D1] optimizer parameters are not exactly render_net'
                  )
          if a3_controller is not None:
              if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'B2':
                  audit_b2_trainable_parameters(
                      model_without_ddp, optimizer_model,
                      output_lr=B2_OUTPUT_LR,
                      gate_lr=float(det_cfg.SELECTIVE_AHDR.GATE_LR),
                  )
              elif os.environ.get('SNORM_EXPERIMENT', 'E0') == 'B1':
                  audit_b1_trainable_parameters(
                      model_without_ddp, optimizer_model,
                      output_lr=B1_OUTPUT_LR,
                      gate_lr=float(det_cfg.SELECTIVE_AHDR.GATE_LR),
                  )
              elif os.environ.get('SNORM_EXPERIMENT', 'E0') == 'A4':
                  audit_a4_trainable_parameters(model_without_ddp, optimizer_model)
              else:
                  audit_a3_trainable_parameters(model_without_ddp, optimizer_model)
          _total_epochs = det_cfg.SOLVER.EPOCHS
          _t_max = max(1, _total_epochs * len(train_loader))
          if os.environ.get('SNORM_EXPERIMENT', 'E0') in ('A4', 'B1', 'B2'):
              _a4_min_factor = 1e-6 / A4_OUTPUT_LR
              def _a4_lr_factor(_step):
                  _progress = min(float(_step), float(_t_max)) / float(_t_max)
                  return _a4_min_factor + 0.5 * (1.0 - _a4_min_factor) * (
                      1.0 + math.cos(math.pi * _progress)
                  )
              schedulers = [
                  torch.optim.lr_scheduler.LambdaLR(
                      optimizer_model, lr_lambda=_a4_lr_factor
                  ),
              ]
          else:
              schedulers = [
                 torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer_model, T_max=_t_max, eta_min=1e-6
                 ),
              ]

          optimizers = [optimizer_model]
          optimizers_evaluation = []

    else:
          if det_cfg.MODEL.MODE.PJOTYPE == 'BN':
              _base = model_without_ddp.teacher
              model_paramters += [{'params': _base.parameters(),'lr':det_cfg.SOLVER.LR}]
          else:
              _base = model_without_ddp.student if hasattr(model_without_ddp, 'student') else model_without_ddp.teacher
              model_paramters += [{'params': _base.backbone.parameters(),'lr':det_cfg.SOLVER.LR}]
              model_paramters += [{'params': _base.rpn.parameters(),'lr':det_cfg.SOLVER.LR}]
              for name, module in _base.box_head.named_children():
                model_paramters += [{'params': module.parameters(),'lr':det_cfg.SOLVER.LR}]
          optimizer_model = torch.optim.Adam(model_paramters, det_cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=det_cfg.SOLVER.WEIGHT_DECAY)
          schedulers = [
             torch.optim.lr_scheduler.MultiStepLR(optimizer_model, det_cfg.SOLVER.STEPS, gamma=det_cfg.SOLVER.GAMMA)
          ]
          optimizers = [optimizer_model]

    losses_writer = None
    if dist_utils.is_main_process():
        losses_writer = SummaryWriter(os.path.join(work_dir, 'losses'))
        losses_writer.add_text('config', '{}'.format(str(det_cfg).replace('\n', '  \n')))
        losses_writer.add_text('args', str(args))

    metrics_writers = {}
    if dist_utils.is_main_process():
        test_dataset_names = [loader.dataset.dataset_name for loader in test_loaders]
        for dataset_name in test_dataset_names:
            metrics_writers[dataset_name] = SummaryWriter(os.path.join(work_dir, 'metrics', dataset_name))

    start_time = time.time()
    epochs = det_cfg.SOLVER.EPOCHS
    global total_steps
    global global_end
    start_epoch = current_epoch + 1
    total_steps = (epochs - start_epoch) * len(train_loader)
    print("Start training, total epochs: {} ({} - {}), total steps: {}".format(epochs - start_epoch, start_epoch, epochs - 1, total_steps))

    ##start memory process
    for epoch in range(start_epoch, epochs):
        if global_end:
              break
              
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
            target_loader.batch_sampler.sampler.set_epoch(epoch)

        def test_func():
            global best_mAP, current_foggy_mAP, current_night_mAP
            global best_sum_mAP, best_step
            updated = False
            model_det.eval()
            metrics = evaluation(model_det, test_loaders, device, det_cfg.TEST.EVAL_TYPES, output_dir=work_dir, iteration=global_step)
            if dist_utils.is_main_process() and losses_writer:
                for dataset_name, metric in metrics.items():
                    for k, v in metric.items():
                        metrics_writers[dataset_name].add_scalar('metrics/' + k, v, global_step=global_step)
                        if k == 'mAP':
                            if 'fog' in dataset_name.lower():
                                current_foggy_mAP = v
                            elif 'night' in dataset_name.lower():
                                current_night_mAP = v
                if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'D1':
                    current_sum = current_foggy_mAP + current_night_mAP
                    if current_sum > best_sum_mAP:
                        best_sum_mAP = current_sum
                        best_mAP = current_sum
                        best_step = global_step
                        updated = True
                        print(
                            '[D1 BEST] step={} fog={:.4f} night={:.4f} '
                            'sum={:.4f}'.format(
                                best_step, current_foggy_mAP,
                                current_night_mAP, current_sum,
                            )
                        )
                else:
                    if current_foggy_mAP > 0.4 and current_night_mAP > 0.4:
                        current_sum = current_foggy_mAP + current_night_mAP
                        if current_sum > best_sum_mAP:
                            best_sum_mAP = current_sum
                            best_mAP = current_sum
                            updated = True
                            print('Both foggy ({:.4f}) and night ({:.4f}) mAP > 0.4, sum improved to {:.4f}'.format(
                                current_foggy_mAP, current_night_mAP, current_sum))
                    else:
                        print('Waiting for both foggy ({:.4f}) and night ({:.4f}) mAP to reach 0.4'.format(
                            current_foggy_mAP, current_night_mAP))
            model_det.train()
            set_d1_frozen_detector_bn_eval(model_det)
            return updated

        def save_func(filename=None, save_str=None):
            state_dict = {
                'model': model_without_ddp.state_dict(),
                #'model_tran': model_tran_without_ddp.state_dict(),
                'current_epoch': epoch,
            }
            filename = filename if filename else 'model_epoch_{:02d}.pth'.format(epoch)
            save_path = os.path.join(work_dir, filename)
            dist_utils.save_on_master(state_dict, save_path)
            if dist_utils.is_main_process() and save_str is not None:
                with open(os.path.join(work_dir, 'best.txt'), 'w') as f:
                    f.write(save_str)

            print('Saved to {}'.format(save_path))

        epoch_start = time.time()
        train_one_epoch(model_det, None, optimizers, schedulers, train_loader, target_loader, device, epoch,
                writer=losses_writer, test_func=test_func, save_func=save_func, det_cfg=det_cfg, args=args,
                a3_controller=a3_controller)

 
        #if epoch> 0 and epoch % 4 ==0:
        if (epoch > -1
                and os.environ.get('SNORM_EXPERIMENT', 'E0') != 'D1'):
            updated = test_func()
            if updated:
                save_func('best.pth', 'sum_mAP: {:.4f} (foggy: {:.4f}, night: {:.4f})'.format(best_mAP, current_foggy_mAP, current_night_mAP))
            print('Best sum mAP: {:.4f} (foggy: {:.4f}, night: {:.4f})'.format(best_mAP, current_foggy_mAP, current_night_mAP))
            
        #if epoch > 10:
        if (epoch > -1
                and os.environ.get('SNORM_EXPERIMENT', 'E0') != 'D1'):
            if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'C3V2':
                save_func('model_epoch_latest.pth')
            else:
                save_func()

        epoch_cost = time.time() - epoch_start
        left = epochs - epoch - 1
        print('Epoch {} ended, cost {}. Left {} epochs, may cost {}'.format(epoch,
                                                                            str(datetime.timedelta(seconds=int(epoch_cost))),
                                                                            left,
                                                                            str(datetime.timedelta(seconds=int(left * epoch_cost)))))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config-file", help="path to config file", type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", help="Only test the model", action="store_true")
    parser.add_argument("--use_current_epoch", help="if use resume epoch", action="store_true")
    parser.add_argument("--config-gan", default='./gan/EFDM/configs/base_train.yaml', help="path to config file", type=str)
    parser.add_argument("--config-save", default='./configs/domain/save/save_MGUIT.yaml', help="path to config file", type=str)
    parser.add_argument("--GAN_TYPE", default='MGUIT', help="types of the transfer model", type=str)
    parser.add_argument('--Gan_TrInterval', default=100000, type=int, help='number of training interval on gan')
    parser.add_argument('--Special_Setting', default=['UNSB'], type=list, help='special types of the transfer model')
    parser.add_argument('--use_i2i', default="train", type=str, help='number of training interval on gan')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", type=int, default=0)
    #parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()
    det_cfg.merge_from_file(args.config_file)
    det_cfg.merge_from_list(args.opts)
    det_cfg.freeze()
    dist_utils.init_distributed_mode(args)

    print(args)
    '''
    world_size = dist_utils.get_world_size()
    if world_size != 4:
        lr = cfg.SOLVER.LR * (float(world_size) / 4)
        print('Change lr from {} to {}'.format(cfg.SOLVER.LR, lr))
        cfg.merge_from_list(['SOLVER.LR', lr])
    '''
    print(det_cfg)
    os.makedirs(det_cfg.WORK_DIR, exist_ok=True)
    if dist_utils.is_main_process():
        with open(os.path.join(det_cfg.WORK_DIR, 'config.yaml'), 'w') as fid:
            fid.write(str(det_cfg))
    
    #signal.signal(signal.SIGTERM, term_sig_hanlder)
    main(det_cfg, args)