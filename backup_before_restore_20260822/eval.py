import os
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import pycocotools.mask as mask_util

from detection import utils
from detection.data.evaluations import coco_evaluation, voc_evaluation
from detection.data.transforms import de_normalize
from detection.layers.mask_ops import paste_masks_in_image
from detection.utils import colormap
from detection.utils.dist_utils import is_main_process, all_gather, get_world_size
from detection.utils.visualizer import Visualizer


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    return predictions


def evaluation(model, data_loaders, device, types=('coco',), output_dir='./evaluations/', iteration=None, viz=False):
    if not isinstance(data_loaders, (list, tuple)):
        data_loaders = (data_loaders,)
    results = dict()
    for data_loader in data_loaders:
        dataset = data_loader.dataset
        _output_dir = os.path.join(output_dir, 'evaluations', dataset.dataset_name)
        os.makedirs(_output_dir, exist_ok=True)
        result = do_evaluation(model, data_loader, device, types=types, output_dir=_output_dir, iteration=iteration, viz=viz)
        results[dataset.dataset_name] = result
    return results


COLORMAP = colormap.colormap(rgb=True, maximum=1)


def save_visualization(dataset, img_meta, result, output_dir, threshold=0.8, fmt='.pdf'):
    save_dir = os.path.join(output_dir, 'visualizations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    file_name = img_meta['img_info']['file_name']
    img = Image.open(os.path.join(dataset.images_dir, file_name))
    w, h = img.size
    scale = 1.0
    # w, h = int(w * scale), int(h * scale)
    # img = img.resize((w, h))

    vis = Visualizer(img, metadata=None)

    boxes = np.array(result['boxes']) * scale
    labels = np.array(result['labels'])
    scores = np.array(result['scores'])
    #masks = result['masks']
    indices = scores > threshold
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    #masks = [m for m, b in zip(masks, indices) if b]

    # colors = COLORMAP[(labels + 2) % len(COLORMAP)]

    colors = [np.array([1.0, 0, 0])] * len(labels)

    labels = ['{}:{:.0f}%'.format(dataset.CLASSES[label], score * 100) for label, score in zip(labels, scores)]
    out = vis.overlay_instances(
        boxes=boxes,
        labels=labels,
        masks=None,
        assigned_colors=colors,
        alpha=0.8,
    )
    out.save(os.path.join(save_dir, os.path.basename(file_name).replace('.', '_') + fmt))


@torch.no_grad()
def do_evaluation(model, data_loader, device, types, output_dir, iteration=None, viz=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    dataset = data_loader.dataset
    header = 'Testing {}:'.format(dataset.dataset_name)
    results_dict = {}
    has_mask = False

    # 渲染结果可视化：每次评估保存前5张图的渲染特征
    render_save_dir = os.path.join(output_dir, 'render_viz')
    os.makedirs(render_save_dir, exist_ok=True)
    render_saved_count = 0

    for images, img_metas, targets, _, _, _, _, _ in metric_logger.log_every(data_loader, 10, header):
        assert len(targets) == 1
        images = images.to(device)
        model_time = time.time()

        # 推理阶段：使用 SNorm_0322_valid_refine.py 中的多尺度融合方法
        det = model(images, img_metas, targets, moe_mode=7)[0]

        # 保存渲染结果可视化（前5张）
        if render_saved_count < 5 and iteration is not None:
            try:
                with torch.no_grad():
                    # 从模型属性获取渲染特征
                    teacher = model.module.teacher if hasattr(model, 'module') else model.teacher
                    features_norm = getattr(teacher, '_last_render_features', None)
                    if features_norm is not None:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                        orig_img = de_normalize(images[0], img_metas[0])
                        axes[0].imshow(orig_img)
                        axes[0].set_title('Input (foggy/night)')
                        axes[0].axis('off')
                        feat = features_norm[0].detach().cpu()
                        if feat.shape[0] >= 3:
                            feat_vis = feat[:3]
                        else:
                            feat_vis = feat.repeat(3, 1, 1)[:3]
                        feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-8)
                        feat_vis = feat_vis.permute(1, 2, 0).numpy()
                        axes[1].imshow(feat_vis)
                        axes[1].set_title('Rendered feature')
                        axes[1].axis('off')
                        plt.tight_layout()
                        save_name = f"render_{dataset.dataset_name}_iter{iteration}_{render_saved_count}.png"
                        plt.savefig(os.path.join(render_save_dir, save_name), dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        render_saved_count += 1
            except Exception:
                pass
            
        boxes, scores, labels = det['boxes'], det['scores'], det['labels']

        model_time = time.time() - model_time

        img_meta = img_metas[0]
        scale_factor = img_meta['scale_factor']
        img_info = img_meta['img_info']

        if viz:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            plt.switch_backend('TKAgg')
            image = de_normalize(images[0], img_meta)
            plt.subplot(122)
            plt.imshow(image)
            plt.title('Predict')
            for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes.tolist(), labels.tolist())):
                if scores[i] > 0.65:
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='g')
                    category_id = dataset.label2cat[label]
                    plt.text(x1, y1, '{}:{:.2f}'.format(dataset.CLASSES[category_id], scores[i]), color='r')
                    plt.gca().add_patch(rect)

            plt.subplot(121)
            plt.imshow(image)
            plt.title('GT')
            for i, ((x1, y1, x2, y2), label) in enumerate(zip(targets[0]['boxes'].tolist(), targets[0]['labels'].tolist())):
                category_id = dataset.label2cat[label]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='g')
                plt.text(x1, y1, '{}'.format(dataset.CLASSES[category_id]))
                plt.gca().add_patch(rect)
            plt.show()

        boxes /= scale_factor
        result = {}

        if 'masks' in det:
            has_mask = True
            (w, h) = img_meta['origin_img_shape']
            masks = paste_masks_in_image(det['masks'], boxes, (h, w))
            rles = []
            for mask in masks.cpu().numpy():
                mask = mask >= 0.5
                mask = mask_util.encode(np.array(mask[0][:, :, None], order='F', dtype='uint8'))[0]
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                mask['counts'] = mask['counts'].decode('utf-8')
                rles.append(mask)
            result['masks'] = rles

        boxes = boxes.tolist()
        # 只在labels是张量时才调用tolist()方法
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        # 确保标签是正确的类别ID
        if hasattr(dataset, 'label2cat'):
            # 确保标签是普通的Python整数，而不是张量
            labels = [dataset.label2cat[label.item() if hasattr(label, 'item') else label] for label in labels]
        scores = scores.tolist()

        result['boxes'] = boxes
        result['scores'] = scores
        result['labels'] = labels

        #save_visualization(dataset, img_meta, result, output_dir, fmt='.jpg')

        results_dict.update({
            img_info['id']: result
        })
        metric_logger.update(model_time=model_time)

    if get_world_size() > 1:
        dist.barrier()

    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
    if not is_main_process():
        return {}
    results = {}
    if has_mask:
        result = coco_evaluation(dataset, predictions, output_dir, iteration=iteration)
        results.update(result)
    if 'voc' in types:
        result = voc_evaluation(dataset, predictions, output_dir, iteration=iteration, use_07_metric=False)
        results.update(result)
    return results
