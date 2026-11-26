import numpy as np
import random
import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as T
import detection.utils.transforms.transforms as T
from detection.utils.layer import GradientReversal
from terminaltables import AsciiTable

from detection.layers import grad_reverse, softmax_focal_loss, sigmoid_focal_loss, style_pool2d, l2_loss
from .backbone import build_backbone
from .roi_heads import BoxHead
from .rpn import RPN

import cv2
import math
import copy
from contextlib import contextmanager

#from detection.modeling.contrastive.GraphContrast import GraphContrastive
from detection.modeling.discriminator.InDiscriminator import InstanceDiscriminator
#from gan.I2IPool_1115 import I2ITienhance
#from detection.modeling.norm.SNorm import SNorm
#from detection.modeling.norm.R202503X.SNorm_0322_god import SNorm
#from detection.modeling.norm.R202503X.SNorm_0322_valid import SNorm
#from detection.modeling.norm.R202503X.SNorm_0322_valid_refine import SNorm
from detection.modeling.norm.R202509x.SNorm_0322_valid_refine import SNorm
from detection.modeling.norm.selective_ahdr import SelectiveAHDRGate
from detection.modeling.norm.detection_benefit import (
    DetectionBenefitPredictor,
    DetectionSemanticBenefitPredictor,
)
from detection.modeling.cdfkd import (
    ConservativeSourceDiversification,
    FeatureResidualAdapter,
    MultiScaleSpatialEnhancementAdapter,
    SpatialEnhancementAdapter,
    global_feature_distillation,
    hard_positive_detection_loss,
    instance_feature_distillation,
    resolve_cdfkd_config,
)
from detection.modeling.advsp import (
    SpectralPerturbationModule, make_spectral_factors,
    update_spectral_factors, fft_backend_name, _tensor_abs_stats,
)
#from detection.modeling.norm.R202509x.SNorm_0322_valid_refine_expmamba import SNorm



@contextmanager
def _advsp_saved_tensor_offload(enabled):
    """Offload max-stage saved activations to CPU without breaking input grads."""
    graph = getattr(torch.autograd, 'graph', None)
    hooks = getattr(graph, 'saved_tensors_hooks', None) if graph is not None else None
    if not enabled or hooks is None:
        yield
        return

    def pack(tensor):
        if not tensor.is_cuda:
            return (None, tensor)
        return (tensor.device, tensor.detach().cpu())

    def unpack(packed):
        device, tensor = packed
        if device is None:
            return tensor
        return tensor.to(device=device)

    with hooks(pack, unpack):
        yield

#
from torch.utils.tensorboard import SummaryWriter
from detection.modeling.norm.script import generate_rect_boxes, select_rect_proposals, select_best_feature, \
generate_rect_boxes_v2, select_rect_proposals_v2, select_rect_weight_v2, select_best_weight, \
select_rect_weight_v3, select_rect_weight_v4, scoring_loss, select_rect_weight_v5, class_loss, \
select_rect_weight_v6, select_rect_weight_v7, select_rect_labels_v7

from detection.modeling.norm.dice_loss import DiceLoss


class ModuleSet(nn.Module):
    def __init__(self, cfg, positive_fraction=0.25, use_eval_head = False, eval_head_num=5, use_memory_conv=False):
        super(ModuleSet, self).__init__()
        self.cfg = cfg
        backbone = build_backbone(cfg)
        in_channels = backbone.out_channels

        #self.backbone = backbone
        
        # 处理 VGG16 backbone
        if cfg.MODEL.BACKBONE.NAME == 'vgg16':
            # VGG16 features 结构: [0-3] 第一个卷积块, [4-7] 第二个卷积块, [8-11] 第三个卷积块, [12-15] 第四个卷积块, [16-19] 第五个卷积块
            self.backbone_0 = backbone[:4]  # 第一个卷积块
            self.backbone_1 = backbone[4:8]  # 第二个卷积块
            self.backbone_2 = backbone[8:12]  # 第三个卷积块
            self.backbone_3 = backbone[12:16]  # 第四个卷积块
            self.backbone_3n = copy.deepcopy(backbone[12:16])
        else:
            # 处理 ResNet backbone
            self.backbone_0 = backbone.stem
            self.backbone_1 = backbone.layer1
            self.backbone_2 = backbone.layer2
            self.backbone_3 = backbone.layer3
            self.backbone_3n = copy.deepcopy(backbone.layer3)
        
        
        #self.rpn = RPN(cfg, in_channels)
        #self.box_head = BoxHead(cfg, in_channels)

        self.rpn_eval = RPN(cfg, in_channels)
        self.box_head_eval = BoxHead(cfg, in_channels)

        self.rpn_norm = RPN(cfg, in_channels)
        self.box_head_norm = BoxHead(cfg, in_channels)

        if cfg.MODEL.MODE.PJOTYPE == 'BN':
            pass
        elif cfg.MODEL.MODE.PJOTYPE == 'NBIAS':
            #base: 0000_1116.log
            #self.proj_norm = nn.BatchNorm2d(512)
            #base: 0000_111x.log
            #self.proj_norm = nn.InstanceNorm2d(512)

            #
            '''
            self.proj_norm = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.ReLU()
            )
            '''
            '''
            self.proj_norm = nn.Sequential(
                nn.InstanceNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU()
            )
            '''
            
            #self.proj_norm = nn.Sequential(
            #    SNorm(),
            #    #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            #    #nn.ReLU()
            #)
            self.proj_norm = SNorm()
            self.structure_guided_ahdr_enabled = bool(
                cfg.C2_STRUCTURE_GUIDANCE.ENABLED
            )
            self.proj_norm.configure_structure_guidance(
                enabled=self.structure_guided_ahdr_enabled,
                cue_type=cfg.C2_STRUCTURE_GUIDANCE.CUE_TYPE,
                strength=cfg.C2_STRUCTURE_GUIDANCE.STRENGTH,
                cue_scale=cfg.C2_STRUCTURE_GUIDANCE.CUE_SCALE,
                blur_kernel=cfg.C2_STRUCTURE_GUIDANCE.BLUR_KERNEL,
            )
            if self.structure_guided_ahdr_enabled:
                print(
                    '[C2 STRUCTURE-GUIDED AHDR] enabled=True '
                    'cue={} strength={} parameters=0'.format(
                        cfg.C2_STRUCTURE_GUIDANCE.CUE_TYPE,
                        cfg.C2_STRUCTURE_GUIDANCE.STRENGTH,
                    )
                )
            self.detection_benefit_enabled = bool(
                cfg.C3_DETECTION_BENEFIT.ENABLED
            )
            if self.detection_benefit_enabled:
                self.detection_benefit_predictor = DetectionBenefitPredictor(
                    channels=3,
                    hidden_channels=cfg.C3_DETECTION_BENEFIT.HIDDEN_CHANNELS,
                    initial_alpha=cfg.C3_DETECTION_BENEFIT.INITIAL_ALPHA,
                )
                print(
                    '[C3 DETECTION-BENEFIT] predictor parameters =',
                    self.detection_benefit_predictor.parameter_count(),
                )
            else:
                self.detection_benefit_predictor = None
            self.detection_semantic_benefit_enabled = bool(
                cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.ENABLED
            )
            if self.detection_semantic_benefit_enabled:
                if int(cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.FEATURE_CHANNELS) != int(in_channels):
                    raise ValueError(
                        'C3-V2 semantic channels mismatch: config=%s backbone=%s'
                        % (
                            cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.FEATURE_CHANNELS,
                            in_channels,
                        )
                    )
                self.detection_semantic_benefit_predictor = (
                    DetectionSemanticBenefitPredictor(
                        in_channels=in_channels,
                        projection_channels=(
                            cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.PROJECTION_CHANNELS
                        ),
                        hidden_channels=(
                            cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.HIDDEN_CHANNELS
                        ),
                        initial_alpha=(
                            cfg.C3V2_DETECTION_SEMANTIC_BENEFIT.INITIAL_ALPHA
                        ),
                    )
                )
                print(
                    '[C3-V2 DETECTION-SEMANTIC BENEFIT] predictor parameters =',
                    self.detection_semantic_benefit_predictor.parameter_count(),
                )
            else:
                self.detection_semantic_benefit_predictor = None
            self.selective_ahdr_enabled = bool(cfg.SELECTIVE_AHDR.ENABLED)
            if self.selective_ahdr_enabled:
                self.selective_ahdr_gate = SelectiveAHDRGate(
                    channels=3,
                    hidden_channels=cfg.SELECTIVE_AHDR.HIDDEN_CHANNELS,
                    alpha_max=cfg.SELECTIVE_AHDR.ALPHA_MAX,
                    gate_init=cfg.SELECTIVE_AHDR.GATE_INIT,
                )
                print(
                    "[B1 SELECTIVE-AHDR] gate parameters =",
                    self.selective_ahdr_gate.parameter_count(),
                )
            else:
                self.selective_ahdr_gate = None

            #self.proj_dbias = I2ITienhance(cfg)
        
        self.ada_layers = [False, True, True]
        self.ada_layers_stride = []
        
        self.llayers_rois = [[0,0.5], [0.2, 0.5], [0.2, 0.5]]
        self.ada_layers_rois = []

        self.positive_fraction = positive_fraction
        self.cdfkd_adapter = None
        self.cdfkd_feature_adapter = None
        self.cdfkd_diversification = None
        self.cdfkd_spectral = None
        self.cdfkd_config = None
        self._cdfkd_identity_checked = False
        if os.environ.get('SNORM_EXPERIMENT', 'E0') == 'CDFKD':
            self.cdfkd_config = resolve_cdfkd_config()
            if self.cdfkd_config['rgb_adapter']:
                cls = (MultiScaleSpatialEnhancementAdapter if
                       self.cdfkd_config['adapter_type'] == 'multiscale'
                       else SpatialEnhancementAdapter)
                self.cdfkd_adapter = cls(self.cdfkd_config['rgb_channels'],
                    self.cdfkd_config['rgb_residual_scale'])
            if self.cdfkd_config['feature_adapter']:
                self.cdfkd_feature_adapter = FeatureResidualAdapter(
                    1024, self.cdfkd_config['feature_channels'],
                    self.cdfkd_config['feature_residual_scale'])
            if self.cdfkd_config.get('source_diversification', True):
                self.cdfkd_diversification = ConservativeSourceDiversification(
                    self.cdfkd_config['div_strength'])
            else:
                print('[CDFKD SOURCE] source_diversification=False mode=original_source_only')
            if self.cdfkd_config.get('advsp', False):
                self.cdfkd_spectral = SpectralPerturbationModule(
                    csg=self.cdfkd_config['advsp_csg'],
                    tsg=self.cdfkd_config['advsp_tsg'],
                    scg=self.cdfkd_config['advsp_scg'],
                    delta_min=self.cdfkd_config['advsp_delta_min'],
                    delta_max=self.cdfkd_config['advsp_delta_max'],
                    object_protect=self.cdfkd_config['advsp_object_protect'],
                    object_rho=self.cdfkd_config['advsp_object_rho'],
                )
                print(
                    '[ADVSP CONFIG] preset={} CSG={} TSG={} SCG={} '
                    'random_only={} r={} steps={} update_tau={} '
                    'delta_min={} delta_max={} mu={} lambda={} '
                    'object_protect={} object_rho={} FFT={}'.format(
                        self.cdfkd_config['preset'],
                        self.cdfkd_config['advsp_csg'],
                        self.cdfkd_config['advsp_tsg'],
                        self.cdfkd_config['advsp_scg'],
                        self.cdfkd_config['advsp_random_only'],
                        self.cdfkd_config['advsp_radius'],
                        self.cdfkd_config['advsp_steps'],
                        self.cdfkd_config['advsp_tau'],
                        self.cdfkd_config['advsp_delta_min'],
                        self.cdfkd_config['advsp_delta_max'],
                        self.cdfkd_config['advsp_mu'],
                        self.cdfkd_config['advsp_lambda'],
                        self.cdfkd_config['advsp_object_protect'],
                        self.cdfkd_config['advsp_object_rho'],
                        fft_backend_name(),
                    )
                )
    
    def prev_norm(self, x):
        
        return x
    
    def back_norm_wo_snorm(self, x):
        #x = self.backbone_2(x)

        x = self.backbone_0(x)
        x = self.backbone_1(x)
        x = self.backbone_2(x)
        x = self.backbone_3(x)
        return x
    
    def back_norm_w_snorm(self, x):
        x = self.backbone_0(x)
        x = self.backbone_1(x)
        x = self.backbone_2(x)
        x = self.backbone_3(x)

        #x = self.backbone_2n(x)
        return x

    def _cdfkd_advsp_detect(self, rendered, img_metas, targets):
        image = (self.cdfkd_adapter(rendered)
                 if self.cdfkd_adapter is not None else rendered)
        feature_base = self.back_norm_w_snorm(image)
        feature = (self.cdfkd_feature_adapter(feature_base)
                   if self.cdfkd_feature_adapter is not None else feature_base)
        proposals, rpn_losses, _, _ = self.rpn_norm(
            image, feature, img_metas, targets
        )
        _, box_losses, _, _, _, _, _, _, _ = self.box_head_norm(
            feature, proposals, img_metas, targets
        )
        return image, feature, rpn_losses, box_losses

    @staticmethod
    def _cdfkd_advsp_det_loss(result):
        return sum(result[2].values()) + sum(result[3].values())

    def _forward_cdfkd_advsp(self, images, img_metas, targets):
        config = self.cdfkd_config
        if self.cdfkd_spectral is None:
            raise RuntimeError('AdvSP spectral module is not initialized')
        # Frozen rendering is used for the clean path; max-stage rendering
        # keeps its input graph so factor gradients reach the detection loss.
        with torch.no_grad():
            clean_rendered, _ = self.dist_prog(
                images.detach(), targets, stride=8, force_eval_render=True,
            )
        # Do not build the trainable clean detector graph before the adversarial
        # max-stage. Keeping that BS=2 graph alive while differentiating through
        # render+detector causes the >24 GiB peak on RTX 3090. The clean detector
        # pass is mathematically independent of factor optimization, so defer it
        # until the ephemeral max-stage graph has been consumed.
        batch = images.size(0)
        factors = make_spectral_factors(
            batch, images.device, images.dtype, config['advsp_radius'],
            config['advsp_csg'], config['advsp_tsg'], config['advsp_scg'],
        )
        # Project metadata stores img_shape as (width, height), while AdvSP
        # consumes valid extents as (height, width). Keep box coordinates unchanged.
        sizes = [
            (int(meta['img_shape'][1]), int(meta['img_shape'][0]))
            for meta in img_metas
        ]
        boxes = [target.get('boxes', target.get('bbox'))
                 if isinstance(target, dict) else target for target in targets]
        sampled_deltas = torch.empty(
            batch, device=images.device, dtype=images.dtype
        ).uniform_(config['advsp_delta_min'], config['advsp_delta_max'])
        factor_names = ('alpha', 'beta', 'gamma')
        factor_diagnostics = {}
        for name, factor in zip(factor_names, factors):
            if factor is not None:
                factor_diagnostics[name] = {
                    'before': _tensor_abs_stats(factor),
                    'grad': _tensor_abs_stats(None),
                    'update': _tensor_abs_stats(None),
                    'after': _tensor_abs_stats(factor),
                    'clamp_fraction': factor.new_zeros(()),
                }
        delta_values = sampled_deltas

        def transform(source):
            return self.cdfkd_spectral(
                source, factors[0], factors[1], factors[2],
                image_sizes=sizes, boxes=boxes, deltas=sampled_deltas,
                return_deltas=True,
            )

        if config.get('advsp_random_only', False):
            adv_image, delta_values = transform(images.detach())
        else:
            for _ in range(config['advsp_steps']):
                spectral, delta_values = transform(images.detach())
                # The renderer and detector are frozen, not graph-disabled:
                # autograd.grad must reach only ephemeral spectral factors.
                # Max-stage needs input gradients all the way back to alpha/beta/gamma,
                # but it does not need to keep those saved activations resident on GPU.
                # CPU saved-tensor offload preserves the exact forward computation and
                # autograd path while reducing the >24 GiB peak seen with BS=2 at 640.
                offload_max_graph = os.environ.get(
                    'SNORM_ADVSP_CPU_OFFLOAD', '1'
                ) != '0'
                with _advsp_saved_tensor_offload(offload_max_graph):
                    adversarial_rendered, _ = self.dist_prog(
                        spectral, targets, stride=8, force_eval_render=True,
                    )
                    adv_result = self._cdfkd_advsp_detect(
                        adversarial_rendered, img_metas, targets
                    )
                    active_factors = [factor for factor in factors if factor is not None]
                    if not active_factors:
                        break
                    gradients = torch.autograd.grad(
                        self._cdfkd_advsp_det_loss(adv_result), active_factors,
                        allow_unused=True,
                    )
                grad_iter = iter(gradients)
                expanded = tuple(None if factor is None else next(grad_iter)
                                 for factor in factors)
                updated = update_spectral_factors(
                    factors, expanded, config['advsp_radius'],
                    config['advsp_tau'],
                )
                for name, factor, gradient, updated_factor in zip(
                        factor_names, factors, expanded, updated):
                    if factor is None:
                        continue
                    record = factor_diagnostics[name]
                    record['grad'] = _tensor_abs_stats(gradient)
                    raw_update = (factor.detach() + config['advsp_tau'] * gradient
                                  if gradient is not None else factor.detach())
                    record['clamp_fraction'] = (
                        ((raw_update > config['advsp_radius']) |
                         (raw_update < -config['advsp_radius']))
                        .to(dtype=factor.dtype).mean().detach()
                    )
                    record['update'] = _tensor_abs_stats(
                        updated_factor - factor.detach()
                    )
                    record['after'] = _tensor_abs_stats(updated_factor)
                factors = tuple(None if factor is None else
                                value.detach().requires_grad_(True)
                                for factor, value in zip(factors, updated))
            adv_image, delta_values = transform(images.detach())

        # Build the persistent clean training graph only after max-stage
        # autograd.grad has consumed and released its temporary graph.
        clean = self._cdfkd_advsp_detect(
            clean_rendered.detach(), img_metas, targets
        )

        # Min stage sees a detached adversarial image; only the persistent
        # feature adapter receives gradients from the second detector pass.
        with torch.no_grad():
            adv_rendered, _ = self.dist_prog(
                adv_image.detach(), targets, stride=8, force_eval_render=True,
            )
        adversarial = self._cdfkd_advsp_detect(
            adv_rendered.detach(), img_metas, targets
        )
        mu = config['advsp_mu']
        advsp_diagnostics = {
            'factors': factor_diagnostics,
            'delta_mean': delta_values.detach().mean(),
            'delta_min': delta_values.detach().min(),
            'delta_max': delta_values.detach().max(),
            'image_diff_mean': (adv_rendered.detach() - clean_rendered.detach()).abs().mean(),
            'image_diff_max': (adv_rendered.detach() - clean_rendered.detach()).abs().max(),
        }
        rpn_losses = {
            key: clean[2][key] + mu * adversarial[2][key]
            for key in clean[2]
        }
        box_losses = {
            key: clean[3][key] + mu * adversarial[3][key]
            for key in clean[3]
        }
        zero = clean[1].sum() * 0.0
        return {
            'rpn_losses': rpn_losses,
            'box_losses': box_losses,
            'reference_feature': clean[1].detach(),
            'student_feature': clean[1],
            'student_image': clean[0],
            'student_base': clean_rendered.detach(),
            'student_feature_base': clean[1],
            'loss_global': zero,
            'loss_instance': zero,
            'loss_hard_object': zero,
            'hard_object_stats': None,
            'diversification': {'operation': 'advsp', 'downscale_ratio': 1.0},
            'advsp_clean_loss': self._cdfkd_advsp_det_loss(clean).detach(),
            'advsp_adv_loss': self._cdfkd_advsp_det_loss(adversarial).detach(),
            'advsp_diagnostics': advsp_diagnostics,
        }

    def forward_cdfkd(self, images, img_metas, targets):
        """Frozen-E2/frozen-detector Stage-A path; only adapter is trainable."""
        if self.cdfkd_config is None:
            raise RuntimeError('CDFKD modules require SNORM_EXPERIMENT=CDFKD')
        if self.cdfkd_config.get('advsp', False):
            return self._forward_cdfkd_advsp(images, img_metas, targets)
        if self.cdfkd_config.get('source_diversification', True):
            if self.cdfkd_diversification is None:
                raise RuntimeError('CDFKD source diversification is not initialized')
            diversified, diversification_stats = self.cdfkd_diversification(images)
        else:
            diversified = images
            diversification_stats = {
                'operation': 'none',
                'downscale_ratio': 1.0,
                'strength': 'original_source_only',
            }
        with torch.no_grad():
            reference_image, _ = self.dist_prog(
                images.detach(), targets, stride=8, force_eval_render=True,
            )
            student_base, _ = self.dist_prog(
                diversified.detach(), targets, stride=8, force_eval_render=True,
            )
            reference_feature = self.back_norm_w_snorm(reference_image).detach()

        student_image = (self.cdfkd_adapter(student_base.detach())
                         if self.cdfkd_adapter is not None else student_base.detach())
        student_feature_base = self.back_norm_w_snorm(student_image)
        student_feature = (self.cdfkd_feature_adapter(student_feature_base)
                           if self.cdfkd_feature_adapter is not None
                           else student_feature_base)
        if not self._cdfkd_identity_checked:
            difference = (student_image.detach() - student_base.detach()).abs()
            mean_difference = float(difference.mean().item())
            max_difference = float(difference.max().item())
            print('[CDFKD IDENTITY CHECK] mean_abs_diff={:.10f} max_abs_diff={:.10f}'.format(
                mean_difference, max_difference
            ))
            if max_difference > 1e-7:
                raise RuntimeError('CDFKD adapter is not identity-initialized')
            if (self.cdfkd_feature_adapter is not None and
                    (student_feature.detach() - student_feature_base.detach()).abs().max().item() > 1e-7):
                raise RuntimeError('CDFKD feature adapter is not identity-initialized')
            self._cdfkd_identity_checked = True

        # Parameters are frozen, but this forward intentionally keeps autograd
        # so detector and FKD losses reach the spatial adapter input.
        proposals, rpn_losses, _, _ = self.rpn_norm(
            student_image, student_feature, img_metas, targets
        )
        _, box_losses, _, _, _, class_logits, labels, _, _ = self.box_head_norm(
            student_feature, proposals, img_metas, targets
        )
        if self.cdfkd_config['hard_object']:
            hard_object_loss, hard_object_stats = hard_positive_detection_loss(
                class_logits, labels,
                min_weight=self.cdfkd_config['hard_min_weight'],
                max_weight=self.cdfkd_config['hard_max_weight'],
            )
        else:
            hard_object_loss = student_feature.sum() * 0.0
            hard_object_stats = None
        loss_global = global_feature_distillation(
            reference_feature, student_feature
        )
        loss_instance = instance_feature_distillation(
            reference_feature, student_feature, self.box_head_norm.pooler,
            targets, img_metas, mode=self.cdfkd_config['instance_mode'],
        )
        return {
            'rpn_losses': rpn_losses,
            'box_losses': box_losses,
            'reference_feature': reference_feature,
            'student_feature': student_feature,
            'student_image': student_image,
            'student_base': student_base.detach(),
            'student_feature_base': student_feature_base,
            'loss_global': loss_global,
            'loss_instance': loss_instance,
            'loss_hard_object': hard_object_loss,
            'hard_object_stats': hard_object_stats,
            'diversification': diversification_stats,
        }
    

    def forward_resnet101(self, x):
        b,c, h, w = x.size()
        adaptation_feats = []
        adaptation_strides = []
        idx = 0
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
            

        idx += 1
        x = self.backbone.layer2(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
        

        idx += 1
        x = self.backbone.layer3(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
        
        self.ada_layers_stride = adaptation_strides

        return x, adaptation_feats
    
    def forward_resnet50(self, x):
        b,c, h, w = x.size()
        adaptation_feats = []
        adaptation_strides = []
        idx = 0
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
            

        idx += 1
        x = self.backbone.layer2(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
        

        idx += 1
        x = self.backbone.layer3(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)
            adaptation_strides.append(w / x.size(-1))
        
        self.ada_layers_stride = adaptation_strides

        return x, adaptation_feats
    
    def forward(self, images, img_metas, targets=None, proposals_=None, \
               is_use_no_grad=False, output_tcs=None, domain='source', use_dbiase=False, \
               results_parameters=None, mode=0, features_ff=None, use_moe=False, use_moe_fusion=False, \
               test_exm=0, moe_mode=0, update_norm_p=False, is_update=False, moe_mark=0):
        outputs = dict()
        #function define        
        
        ##rpn and rcnn
        if mode== 0: #score training/evaluation
            # 对于默认模式，使用正常的前向传播流程
            features_prev = self.prev_norm(images)
            # 通过 dist_prog 处理特征，传递 is_night 参数
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode, moe_mark=moe_mark)
            features_ff = self.back_norm_w_snorm(features_norm)
            features_ff = features_ff.detach()
            proposals, rpn_losses, rpn_logits, rpn_boxes = self.rpn_eval(images, features_ff, img_metas, targets)
            
            ##
            dets, box_losses, _, _, _, _, _, obj_dets, _ = \
            self.box_head_eval(features_ff, proposals, img_metas, targets, results_parameters=results_parameters, test_exm=test_exm)
            dets_bb = None
            features = None
            outputs['proposals'] = proposals
            outputs['rpn_losses_eval'] = rpn_losses
            outputs['box_losses_eval'] = box_losses
            outputs['dets_eval'] = dets
            outputs['dets_w'] = dets

        elif mode== 1: #detection training/evaluation
            features_prev = self.prev_norm(images)

            #without norm
            '''
            features_wo = self.back_norm_wo_snorm(features_prev)
            proposals_wo, rpn_losses_wo, _, _ = self.rpn(images, features_wo, img_metas, targets)
            dets, box_losses_wo, _, _, _, _, _, obj_dets, _ = self.box_head(features_wo, proposals_wo, img_metas, targets)
            '''

            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            #if update_norm_p:
            #    self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
                
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode, moe_mark=moe_mark)
            if (os.environ.get('SNORM_EXPERIMENT', 'E0') == 'CDFKD'
                    and self.cdfkd_config is not None):
                if self.cdfkd_adapter is not None:
                    features_norm = self.cdfkd_adapter(features_norm)
            # 保存渲染结果用于评估可视化
            self._last_render_features = features_norm.detach()
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            if self.cdfkd_feature_adapter is not None:
                features_w = self.cdfkd_feature_adapter(features_w)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            '''
            if update_norm_p:
                loss_weights_0 = sum(list(rpn_losses_w.values()))
                loss_weights_1 = sum(list(box_losses_w.values()))
                loss_weights = loss_weights_0 + loss_weights_1
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update, loss_weights=loss_weights)
            '''

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev
        
            
        
        elif mode== 2: #fusion training
            #with norm
            features_norm, feat_dict = self.dist_prog(features_ff.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)

            '''
            features_w = self.back_norm_w_snorm(features_norm)

            mproposals, rpn_losses_m, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            _, box_losses_m, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, mproposals, img_metas, targets)
            outputs['rpn_losses_m'] = rpn_losses_m
            outputs['box_losses_m'] = box_losses_m
            dets_bb = None
            outputs['features_norm'] = features_norm
            outputs['features_w'] = features_w
            '''
            
            
            rpn_losses_m = 0.0
            box_losses_m = 0.0
            
            for idx_ii in range(len(feat_dict['feature_dict_per'])):
                win_sz = feat_dict['win_sz_set'][idx_ii]
                features_norm_l = feat_dict['feature_dict_per'][idx_ii]

                nimages, ntargets, nimg_metas, nfeatures_norm_l = select_rect_labels_v7(images, targets, img_metas, \
                features_norm_l, win_sz_set=win_sz)

                if len(ntargets) == 0:
                    continue
                
                features_w_l = self.back_norm_w_snorm(nfeatures_norm_l)
                mproposals_l, rpn_losses_ml, _, _ = self.rpn_norm(nimages, features_w_l, nimg_metas, ntargets)
                _, box_losses_ml, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w_l, mproposals_l, \
                                                                nimg_metas, ntargets)
                rpn_losses_m += sum(rpn_losses_ml.values())
                box_losses_m += sum(box_losses_ml.values())
                #print(rpn_losses_ml)
            
            '''
            for idx_ii in range(len(feat_dict['feature_dict_fused'])):
                features_norm_l = feat_dict['feature_dict_fused'][idx_ii]
                features_w_l = self.back_norm_w_snorm(features_norm_l)
                mproposals_l, rpn_losses_ml, _, _ = self.rpn_norm(images, features_w_l, img_metas, targets)
                _, box_losses_ml, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w_l, mproposals_l, img_metas, targets)
                rpn_losses_m += sum(rpn_losses_ml.values())
                box_losses_m += sum(box_losses_ml.values())
            '''
            
                
            
            outputs['rpn_losses_m'] = rpn_losses_m
            outputs['box_losses_m'] = box_losses_m
            outputs['features_norm'] = features_norm
            outputs['dets_w'] = dets_w
        
        elif mode== 3: #eval_Net training
            with torch.no_grad():
                features_prev = self.prev_norm(images)
                #with norm
                if update_norm_p:
                    self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
                
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)

            with torch.no_grad():
                #features_w = self.back_norm_w_snorm(features_norm)
                
                dets_w_set = []
                len_num = len(feat_dict['feature_dict'])
                results_parameters = {'score_thresh':0.3, 'nms_thresh':0.6}
                for idx_num in range(len_num):
                    features_w = self.back_norm_w_snorm(feat_dict['feature_dict'][idx_num])
                    proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
                    dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets, results_parameters=results_parameters)
                    dets_w_set.append(dets_w)
            
            outputs['dets_w'] = dets_w_set
        
        ####detection training####
        elif mode== 4: 
            features_prev = self.prev_norm(images)

            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            if update_norm_p:
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
            
            #print('----1')
            #features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            #use_dbiase=use_dbiase, moe_mode=moe_mode)
            features_norm, feat_dict = features_prev, None
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev
        
        elif mode== 5: #detection training
            features_prev = self.prev_norm(images)

            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            if update_norm_p:
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
            
            features_norm, feat_dict = self.dist_prog(
                features_prev.detach(), targets, stride=8, domain=domain,
                use_dbiase=use_dbiase, moe_mode=moe_mode,
                # D4 also needs the deterministic/eval-style legacy E2
                # render path for frozen-detector task guidance.  Unlike
                # D1/D2/D3, D4 does not backpropagate through render_net,
                # so reentrant AHDR checkpointing remains disabled for D4.
                force_eval_render=(os.environ.get('SNORM_EXPERIMENT', 'E0') in ('D1', 'D2', 'D3', 'D4')),
                d1_det_checkpoint=(os.environ.get('SNORM_EXPERIMENT', 'E0') in ('D1', 'D2', 'D3')),
            )
            #features_norm, feat_dict = features_prev, None
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev
        
        #####e-net####
        elif mode== 6: 
            features_prev = self.prev_norm(images)

            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            if update_norm_p:
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
            
            #print('----1')
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)
            #features_norm, feat_dict = features_prev, None
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev

        elif mode== 7: 
            features_prev = self.prev_norm(images)
            dets = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #print('----1')
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode)
            #features_norm, feat_dict = features_prev, None
            #features_norm = features_prev
            #feat_dict = None
            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(images, features_w, img_metas, targets)
            dets_w, box_losses_w, _, _, _, _, _, obj_dets, _ = self.box_head_norm(features_w, proposals_w, img_metas, targets)
            dets_bb = None

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_wo'] = rpn_losses_wo
            outputs['box_losses_wo'] = box_losses_wo
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev

        elif mode== 8: 
            features_prev = self.prev_norm(images)

            dets = None
            dets_w = None
            features_wo = None
            rpn_losses_wo = None
            box_losses_wo = None

            #with norm
            if update_norm_p:
                self.proj_norm.update_norm_parameters(features_prev, is_update=is_update)
            
            #print('----1')
            features_norm, feat_dict = self.dist_prog(features_prev.detach(), targets, stride=8, domain=domain, \
            use_dbiase=use_dbiase, moe_mode=moe_mode, moe_mark=moe_mark)

            outputs['dets_wo'] = dets
            outputs['dets_w'] = dets_w
            outputs['features_wo'] = features_wo
            outputs['features_norm'] = features_norm
            if feat_dict is not None:
                if 'attn' in feat_dict.keys():
                    outputs['attn'] = feat_dict['attn']
                if 'attn_masks' in feat_dict.keys():
                    outputs['attn_masks'] = feat_dict['attn_masks']

        elif mode == 9:  # C3 detection-benefit-aware training
            if not self.detection_benefit_enabled:
                raise RuntimeError('ModuleSet mode=9 requires C3 predictor')
            features_prev = self.prev_norm(images)
            features_norm, feat_dict = self.dist_prog(
                features_prev.detach(), targets, stride=8, domain=domain,
                use_dbiase=use_dbiase, moe_mode=moe_mode,
                moe_mark=moe_mark,
            )
            base = feat_dict['c3']['base']
            ahdr = feat_dict['c3']['ahdr']
            base_losses, ahdr_losses = self._paired_detection_losses(
                images, base, ahdr, img_metas, targets
            )

            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(
                images, features_w, img_metas, targets
            )
            dets_w, box_losses_w, _, _, _, _, _, _, _ = self.box_head_norm(
                features_w, proposals_w, img_metas, targets
            )
            outputs['dets_w'] = dets_w
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev
            outputs['c3'] = feat_dict['c3']
            outputs['c3']['base_detection_loss'] = base_losses
            outputs['c3']['ahdr_detection_loss'] = ahdr_losses
            outputs['c3']['benefit'] = (base_losses - ahdr_losses).detach()

        elif mode == 10:  # C3-V2 semantic benefit training
            if not self.detection_semantic_benefit_enabled:
                raise RuntimeError('ModuleSet mode=10 requires C3-V2 predictor')
            features_prev = self.prev_norm(images)
            features_norm, feat_dict = self.dist_prog(
                features_prev.detach(), targets, stride=8, domain=domain,
                use_dbiase=use_dbiase, moe_mode=moe_mode,
                moe_mark=moe_mark,
            )
            c3v2 = feat_dict['c3v2']
            base_losses, ahdr_losses = (
                self._paired_detection_losses_from_features(
                    images,
                    c3v2['base_semantic_feature'],
                    c3v2['ahdr_semantic_feature'],
                    img_metas,
                    targets,
                )
            )

            features_w = self.back_norm_w_snorm(features_norm)
            proposals_w, rpn_losses_w, _, _ = self.rpn_norm(
                images, features_w, img_metas, targets
            )
            dets_w, box_losses_w, _, _, _, _, _, _, _ = self.box_head_norm(
                features_w, proposals_w, img_metas, targets
            )
            outputs['dets_w'] = dets_w
            outputs['features_w'] = features_w
            outputs['features_norm'] = features_norm
            outputs['rpn_losses_w'] = rpn_losses_w
            outputs['box_losses_w'] = box_losses_w
            outputs['features_prev'] = features_prev
            outputs['c3v2'] = c3v2
            outputs['c3v2']['base_detection_loss'] = base_losses
            outputs['c3v2']['ahdr_detection_loss'] = ahdr_losses
            outputs['c3v2']['benefit'] = (
                base_losses - ahdr_losses
            ).detach()

        outputs['feat_dict'] = feat_dict
            
        return outputs
    
    def dist_prog(self, x, targets, stride=8, use_dbiase=False, domain='source', use_moe=False, use_moe_fusion=False, moe_mode=0, moe_mark=0, force_eval_render=False, d1_det_checkpoint=False):
        if self.cfg.MODEL.MODE.PJOTYPE == 'BN':
            _empty = {
                'feature_dict': [],
                'feature_dict_per': [],
                'feature_dict_fused': [],
                'weight_dict': [],
                'win_sz_set': [],
                'attn': [],
                'attn_masks': None,
            }
            return x, _empty
        elif self.cfg.MODEL.MODE.PJOTYPE == 'NBIAS':
            nx, feat_dict = self.proj_norm(
                x,
                use_moe=use_moe,
                use_moe_fusion=use_moe_fusion,
                moe_mode=moe_mode,
                moe_mark=moe_mark,
                targets=targets,
                force_eval_render=(
                    force_eval_render or (
                        (self.selective_ahdr_enabled
                         or self.structure_guided_ahdr_enabled
                         or self.detection_benefit_enabled
                         or self.detection_semantic_benefit_enabled)
                        and self.training
                    )
                ),
                d1_det_checkpoint=d1_det_checkpoint,
            )
            if self.detection_benefit_enabled:
                required = ('selective_base', 'selective_legacy')
                missing = [key for key in required if key not in feat_dict]
                if missing:
                    raise RuntimeError(
                        'C3 AHDR endpoints missing: %s' % (missing,)
                    )
                base = feat_dict['selective_base']
                ahdr = feat_dict['selective_legacy']
                alpha, logits, descriptor = self.detection_benefit_predictor(
                    base, ahdr
                )
                # Final detection does not train the selector.  Only the
                # detached, paired detection-benefit target trains it.
                nx = base + alpha.detach() * (ahdr - base)
                feat_dict['c3'] = {
                    'base': base,
                    'ahdr': ahdr,
                    'selected': nx,
                    'alpha': alpha,
                    'logits': logits,
                    'descriptor': descriptor,
                }
            if self.detection_semantic_benefit_enabled:
                required = ('selective_base', 'selective_legacy')
                missing = [key for key in required if key not in feat_dict]
                if missing:
                    raise RuntimeError(
                        'C3-V2 AHDR endpoints missing: %s' % (missing,)
                    )
                base = feat_dict['selective_base']
                ahdr = feat_dict['selective_legacy']
                # Reused by the benefit RPN/ROI branches below.  No additional
                # base/AHDR backbone pass is performed in mode=10.
                with torch.no_grad():
                    base_semantic = self.back_norm_w_snorm(base.detach())
                    ahdr_semantic = self.back_norm_w_snorm(ahdr.detach())
                alpha, logits, descriptor = (
                    self.detection_semantic_benefit_predictor(
                        base_semantic, ahdr_semantic
                    )
                )
                nx = base + alpha.detach() * (ahdr - base)
                c3v2_payload = {
                    'base': base,
                    'ahdr': ahdr,
                    'selected': nx,
                    'alpha': alpha,
                    'logits': logits,
                    'descriptor': descriptor,
                }
                if self.training:
                    c3v2_payload['base_semantic_feature'] = (
                        base_semantic.detach()
                    )
                    c3v2_payload['ahdr_semantic_feature'] = (
                        ahdr_semantic.detach()
                    )
                feat_dict['c3v2'] = c3v2_payload
            if self.selective_ahdr_enabled:
                required = ("selective_base", "selective_restored", "selective_legacy")
                missing = [key for key in required if key not in feat_dict]
                if missing:
                    raise RuntimeError(
                        "B1 Selective-AHDR endpoints missing: %s" % missing
                    )
                nx, selective = self.selective_ahdr_gate(
                    feat_dict["selective_base"],
                    feat_dict["selective_restored"],
                )
                feat_dict["selective"] = selective
                feat_dict["initialization_diff"] = (
                    nx - feat_dict["selective_legacy"]
                )
            if  use_dbiase:
                domain_num = 0 if domain == 'source' else 1
                nx = self.proj_dbias.read_memory(nx, targets, stride=stride, domain=1)
                self.pjo_ops.update_memory(x, targets, stride=stride, domain=1)
            return nx, feat_dict

    @staticmethod
    def _capture_rng_state(images):
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': None,
        }
        if images.is_cuda:
            state['cuda'] = torch.cuda.get_rng_state(images.device)
        return state

    @staticmethod
    def _restore_rng_state(state, images):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if images.is_cuda and state['cuda'] is not None:
            torch.cuda.set_rng_state(state['cuda'], images.device)

    def _single_detection_loss(self, images, representation,
                               img_metas, targets):
        features = self.back_norm_w_snorm(representation)
        proposals, rpn_losses, _, _ = self.rpn_norm(
            images, features, img_metas, targets
        )
        _, box_losses, _, _, _, _, _, _, _ = self.box_head_norm(
            features, proposals, img_metas, targets
        )
        return sum(rpn_losses.values()) + sum(box_losses.values())

    def _paired_detection_losses(self, images, base, ahdr,
                                 img_metas, targets):
        """Per-sample base/AHDR loss with identical proposal sampling RNG."""
        base_values = []
        ahdr_values = []
        with torch.no_grad():
            for index in range(images.size(0)):
                sample_images = images[index:index + 1]
                sample_meta = img_metas[index:index + 1]
                sample_targets = targets[index:index + 1]
                rng_state = self._capture_rng_state(sample_images)
                base_loss = self._single_detection_loss(
                    sample_images, base[index:index + 1].detach(),
                    sample_meta, sample_targets,
                )
                self._restore_rng_state(rng_state, sample_images)
                ahdr_loss = self._single_detection_loss(
                    sample_images, ahdr[index:index + 1].detach(),
                    sample_meta, sample_targets,
                )
                base_values.append(base_loss.detach())
                ahdr_values.append(ahdr_loss.detach())
        return torch.stack(base_values), torch.stack(ahdr_values)

    def _single_detection_loss_from_feature(self, images, feature,
                                            img_metas, targets):
        proposals, rpn_losses, _, _ = self.rpn_norm(
            images, feature, img_metas, targets
        )
        _, box_losses, _, _, _, _, _, _, _ = self.box_head_norm(
            feature, proposals, img_metas, targets
        )
        missing_rpn = {
            'rpn_cls_loss', 'rpn_reg_loss'
        } - set(rpn_losses.keys())
        missing_roi = {
            'rcnn_cls_loss', 'rcnn_reg_loss'
        } - set(box_losses.keys())
        if missing_rpn or missing_roi:
            raise RuntimeError(
                'C3-V2 detection benefit losses missing: rpn=%s roi=%s'
                % (sorted(missing_rpn), sorted(missing_roi))
            )
        return sum(rpn_losses.values()) + sum(box_losses.values())

    def _paired_detection_losses_from_features(
            self, images, base_feature, ahdr_feature, img_metas, targets):
        """C3-V2 loss labels reusing already-computed semantic features."""
        base_values = []
        ahdr_values = []
        with torch.no_grad():
            for index in range(images.size(0)):
                sample_images = images[index:index + 1]
                sample_meta = img_metas[index:index + 1]
                sample_targets = targets[index:index + 1]
                rng_state = self._capture_rng_state(sample_images)
                base_loss = self._single_detection_loss_from_feature(
                    sample_images, base_feature[index:index + 1],
                    sample_meta, sample_targets,
                )
                self._restore_rng_state(rng_state, sample_images)
                ahdr_loss = self._single_detection_loss_from_feature(
                    sample_images, ahdr_feature[index:index + 1],
                    sample_meta, sample_targets,
                )
                base_values.append(base_loss.detach())
                ahdr_values.append(ahdr_loss.detach())
        return torch.stack(base_values), torch.stack(ahdr_values)
    
    def compute_targets_for_locations(self, locations, targets, feats):
        bn, lc, lh, lw = feats.size()
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        #print(locations.size(), len(targets))
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im['boxes']
            labels_per_im = targets_per_im['labels']
            
            if bboxes.size(0) == 0:
                labels_per_im = torch.zeros_like(locations[:,0])
                
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
                
                #print('--0:', labels_per_im.size())
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
        
        #print(locations.size(), len(targets),feats.size())
        labels = torch.stack(labels, dim=0)
        labels = labels.view(bn, lh, lw)
        
        reg_targets = torch.stack(reg_targets, dim=0)
        reg_targets = reg_targets.view(bn, lh, lw, 4)
        
        return labels, reg_targets
    
    def compute_locations(self, features, fpn_strides=[8]):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
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
    
    def tranformer_target3(self, features, img_metas, targets_, vis=False):
        targetn_ = []
        nfeatures = []
        nimg_metas = []
        
        for idx in range(len(targets_)):
            target_ = targets_[idx]

            if target_['labels'].size(0) == 0:
                continue
            
            bbs = target_['boxes'].clone().detach()
            lls = target_['labels'].clone().detach()
            
            if len(bbs.size()) == 2 and bbs.size(0) > 0:
                #nimages.append(images[idx])
                nfeatures.append(features[idx])
                nimg_metas.append(img_metas[idx])
                targetn_.append({'boxes':bbs, 'labels': lls})
        
        if len(nfeatures) > 0:
            #nimages = torch.stack(nimages, dim=0)
            nfeatures = torch.stack(nfeatures, dim=0)
            return nfeatures, nimg_metas, targetn_
        else:
            return features, img_metas, None
