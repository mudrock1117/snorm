import argparse
import datetime
import math
import os
import time
import copy
import random
import signal
import numpy as np

# CUDA预初始化：在任何其他导入之前初始化CUDA，防止tensorflow导入失败污染CUDA上下文
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# 降低显存碎片化风险：限制单次分配上限，避免大块分裂
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
import torch
torch.cuda.init()  # 显式初始化CUDA上下文
torch.cuda.synchronize()  # 确保CUDA就绪
import torchvision
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from detection import utils
from detection.config import cfg as det_cfg
from detection.data.build import build_data_loaders
from detection.engine.eval import evaluation
from detection.modeling.build import build_detectors
from detection.utils import dist_utils

#from saver_domain import Saver
import torch.nn.functional as F
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

def term_sig_hanlder(signum, frame):
    global global_end
    global_end = True


def train_one_epoch(model_det, model_tran, optimizers, lr_schedulers, train_loader, target_loader, device, epoch, print_freq=10, writer=None, test_func=None, save_func=None, \
det_cfg=None, args=None, i2i_cfg=None, save_cfg=None, eval_model=None, saver=None):
    global global_step
    global global_start
    global global_index
    global global_end
    global sample_index_iter
    
    global global_step_gan
    global is_update_memory
    
    model_det.train()

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
            '''
            #mode=0, detector and gated network
            loss_dict_s0, outputs_s = model_det(tr_images, img_metas, s_targets, mode_training=7, step_iter=global_step, \
                                                is_update=is_update, stsc_images=sc_images, stsc_trans=sc_trans)
            det_loss_s0 = sum(list(loss_dict_s0.values()))
            det_loss_s0.backward()
            metric_logger.update(det_loss_s=det_loss_s0)
            del det_loss_s0, loss_dict_s0, outputs_s
            torch.cuda.empty_cache()

            #mode=3, distru
            loss_dict_s2, _ = model_det(tr_images, img_metas, s_targets, mode_training=8, outputs_teacher=None, \
                                        stsc_images=sc_images, stsc_trans=sc_trans, step_iter=global_step)
            det_loss_s2 = sum(list(loss_dict_s2.values()))
            det_loss_s2.backward()
   
            metric_logger.update(det_loss_s2=det_loss_s2)
            del det_loss_s2, loss_dict_s2
            torch.cuda.empty_cache()
            '''

            ###image render###

            loss_dict_s3, _ = model_det(images, img_metas, s_targets, mode_training=10, outputs_teacher=None, \
                                        moe_mark=0, stpd_images=tr_images, stpd_trans=tr_trans, step_iter=global_step)

            det_loss_s3 = 100 * sum(list(loss_dict_s3.values()))
            det_loss_s3.backward()

            ##################
            torch.nn.utils.clip_grad_norm_(model_det.parameters(), max_norm=1.0)

            '''
            loss_dict_s4, _ = model_det(images, img_metas, s_targets, mode_training=10, outputs_teacher=None, \
                                        moe_mark=1, stpd_images=tr_images, stpd_trans=tr_trans, step_iter=global_step)

            det_loss_s4 = 100 * sum(list(loss_dict_s4.values()))
            det_loss_s4.backward()

            '''
            loss_dict_s5, _ = model_det(images, img_metas, s_targets, mode_training=11, outputs_teacher=None, \
                                        moe_mark=2, stpd_images=tr_images, stpd_trans=tr_trans, step_iter=global_step)

            det_loss_s5 = 100 * sum(list(loss_dict_s5.values()))
            det_loss_s5.backward()


            #loss_dict_s5, _ = model_det(images, img_metas, s_targets, mode_training=10, outputs_teacher=None, \
            #                            moe_mark=2, stpd_images=tr_images, stpd_trans=tr_trans, step_iter=global_step)
            #
            #det_loss_s5 = 100 * sum(list(loss_dict_s5.values()))
            #det_loss_s5.backward()
            #torch.nn.utils.clip_grad_norm_(model_det.parameters(), 0.5)
            #metric_logger.update(det_loss_s1=det_loss_s3)
            metric_logger.update(det_loss_s=det_loss_s3)
            metric_logger.update(det_loss_s3=det_loss_s3)
            loss_attn_val = loss_dict_s3.get('loss_attn', torch.tensor(0.0))
            metric_logger.update(det_loss_attn=loss_attn_val)
            
            #del det_loss_s3, loss_dict_s3
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            ##################

            for optimizer in optimizers:
                optimizer.step()
        
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

        if global_step % (1000 // max(1, (dist_utils.get_world_size() // 2))) == 0 and test_func is not None:
            updated = test_func()
            if updated:
                save_func('best.pth', 'sum_mAP: {:.4f} (foggy: {:.4f}, night: {:.4f})'.format(best_mAP, current_foggy_mAP, current_night_mAP))
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
    for name, m in snorm_modules:
        m._clip_internal_params()
        if m.check_nan():
            print(f"[Epoch {epoch}] WARNING: SNorm '{name}' detected NaN/Inf in alpha params, resetting EMA...")
            m.reset_ema()
        print(f"[Epoch {epoch}] SNorm '{name}': alpha_ba={m.alpha_ba.data.cpu().numpy()}, alpha_bu={m.alpha_bu.data.cpu().numpy()}")

    if hasattr(model_det, 'module'):
        model_det.module.teacher.proj_norm.compute_norm_parameters(device=device)
    else:
        model_det.teacher.proj_norm.compute_norm_parameters(device=device)

def main(det_cfg, args):
    print('----num_workers:', det_cfg.DATALOADER.NUM_WORKERS)
    
    # 只在非测试模式下加载训练数据
    if not args.test_only:
        train_loader = build_data_loaders(det_cfg.DATASETS.TRAINS, transforms=det_cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                      batch_size=det_cfg.SOLVER.BATCH_SIZE, num_workers=det_cfg.DATALOADER.NUM_WORKERS, shuffle=False)
        target_loader = build_data_loaders(det_cfg.DATASETS.TARGETS, transforms=det_cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                       batch_size=det_cfg.SOLVER.BATCH_SIZE, num_workers=0, shuffle=True)
    
    # 加载测试数据
    test_loaders = build_data_loaders(det_cfg.DATASETS.TESTS, transforms=det_cfg.INPUT.TRANSFORMS_TEST, is_train=False,
                                      distributed=args.distributed, num_workers=0)
    
    ###load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_det = build_detectors(det_cfg)
    model_det.to(device)
    
    model_without_ddp = model_det
    model_tran_without_ddp = None
    if args.distributed:
        print('--set distributed')
        model_det = DistributedDataParallel(convert_sync_batchnorm(model_det), device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False)
       
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
        model_without_ddp = model_det.module
    
    current_epoch = -1
    if args.resume:
        print('Loading from {} ...'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        # 加载完整权重，只忽略不匹配的参数
        # 这样可以确保所有兼容的参数都被加载，而不是过滤掉部分参数
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print('✅ 成功加载权重参数')
        if 'current_epoch' in checkpoint and args.use_current_epoch:
            current_epoch = int(checkpoint['current_epoch'])
    
    work_dir = det_cfg.WORK_DIR
    if args.test_only:
        evaluation(model_det, test_loaders, device, types=det_cfg.TEST.EVAL_TYPES, output_dir=work_dir)
        return
    
    ##set optimizer##
    print('--set optimizer')
    model_paramters = []
    model_eval_parameters = []
    if det_cfg.MODEL.MODE.TRAIN_PROCESS == 'S0':
          model_paramters += [{'params': model_without_ddp.teacher.proj_norm.render_net.parameters(),'lr':det_cfg.SOLVER.LR}]
          model_paramters += [{'params': model_without_ddp.dis.parameters(),'lr':det_cfg.SOLVER.LR}]
          optimizer_model = torch.optim.Adam(model_paramters, det_cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=det_cfg.SOLVER.WEIGHT_DECAY)
          _total_epochs = det_cfg.SOLVER.EPOCHS
          _t_max = max(1, _total_epochs * len(train_loader))
          schedulers = [
             torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=_t_max, eta_min=1e-6),
          ]

          optimizers = [optimizer_model]
          optimizers_evaluation = []

    else:
          model_paramters += [{'params': model_without_ddp.student.backbone.parameters(),'lr':det_cfg.SOLVER.LR}]
          model_paramters += [{'params': model_without_ddp.student.rpn.parameters(),'lr':det_cfg.SOLVER.LR}]
          for name, module in model_without_ddp.student.box_head.named_children():
            model_paramters += [{'params': module.parameters(),'lr':det_cfg.SOLVER.LR}]

          optimizer_model = torch.optim.Adam(model_paramters, det_cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=det_cfg.SOLVER.WEIGHT_DECAY)
          schedulers = [
             torch.optim.lr_scheduler.MultiStepLR(optimizer_model, det_cfg.SOLVER.STEPS, gamma=det_cfg.SOLVER.GAMMA),
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
            global best_mAP, current_foggy_mAP, current_night_mAP, best_sum_mAP
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
                writer=losses_writer, test_func=test_func, save_func=save_func, det_cfg=det_cfg, args=args)

 
        #if epoch> 0 and epoch % 4 ==0:
        if epoch > -1:
            updated = test_func()
            if updated:
                save_func('best.pth', 'sum_mAP: {:.4f} (foggy: {:.4f}, night: {:.4f})'.format(best_mAP, current_foggy_mAP, current_night_mAP))
            print('Best sum mAP: {:.4f} (foggy: {:.4f}, night: {:.4f})'.format(best_mAP, current_foggy_mAP, current_night_mAP))
            
        #if epoch > 10:
        if epoch > -1:
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