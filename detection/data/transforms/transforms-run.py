import random

import cv2
import numpy as np
import torch

from detection.data.container import Container
from detection.structures import PolygonMasks
from .aug2_utils import *

class random_flip(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def flip_boxes(self, results):
        if 'boxes' in results:
            w, h = results['img_shape']
            boxes = results['boxes']
            flipped = boxes.copy()
            flipped[..., 0] = w - boxes[..., 2] - 1
            flipped[..., 2] = w - boxes[..., 0] - 1
            results['boxes'] = flipped

    def flip_masks(self, results):
        if 'masks' in results:  # list[list[ndarray[double]]]
            w, h = results['img_shape']
            masks = results['masks']
            for mask in masks:
                for polygon in mask:
                    # np.array([x0, y0, x1, y1, ..., xn, yn]) (n >= 3)
                    polygon[0::2] = w - polygon[0::2] - 1
            results['masks'] = masks

    def __call__(self, results):
        #print('--0')
        flip = True if np.random.rand() < self.flip_ratio else False
        results['flip'] = flip
        if results['flip']:
            results['img'] = np.flip(results['img'], axis=1)
            self.flip_boxes(results)
            self.flip_masks(results)
        return results


class resize(object):
    def __init__(self, min_size, max_size=None, keep_ratio=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio

    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return w, h
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return ow, oh

    def resize_boxes(self, results):
        w, h = results['img_shape']
        boxes = results['boxes'].copy() * results['scale_factor']
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
        results['boxes'] = boxes

    def resize_masks(self, results):
        if 'masks' in results:
            masks = results['masks']
            for mask in masks:
                for polygon in mask:
                    scale_y = scale_x = results['scale_factor']
                    # inplace modify
                    polygon[0::2] *= scale_x
                    polygon[1::2] *= scale_y
            results['masks'] = masks

    def __call__(self, results):
        w, h = results['img_shape']
        new_w, new_h = self.get_size((w, h))
        image = cv2.resize(results['img'], dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

        results['img'] = image

        results['img_shape'] = (new_w, new_h)
        results['origin_img_shape'] = (w, h)
        results['scale_factor'] = float(new_w) / w

        self.resize_boxes(results)
        self.resize_masks(results)
        #print('--1')

        return results


class normalize(object):
    """Normalize the image.
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_bgr (bool): Whether to convert the image from RGB to BGR,
            default is true.
    """

    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1), to_01=False, to_bgr=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_01 = to_01
        self.to_bgr = to_bgr

    def __call__(self, results):
        #print('--2')
        #print(results['img'].shape)
        RESIZE = results['img'].shape[1]
        SCALE_SZ0 = [0.5, 2.0]#[0.5, 1.5]#[0.25, 2.0]#[0.25, 2.0]
        SCALE_SZ1 = [0.6, 1.4]
        #SCALE_SZ1 = [1.0001, 1.0002]
        RCROP = [results['img'].shape[0], results['img'].shape[1]]
        
        ##scale
        aga = 0.3#random.random() * 0.3
        asn = 0.3#random.random() * 0.3
        ahe = 0.3#random.random() * 0.3
        act = 0.3#random.random() * 0.3
        rhf = 0.2#0.4
        rvf = 0.2#0.4
        
        
        #original
        if results['is_sample'] == 2:
            #SCALE_SZ2 = [1.0, 2.0, 3.0]
            SCALE_SZ2 = [1.0, 1.5]
            #45.6， 46.6， 47.5
            '''
            aug_orignal_txt = Compose([AdjustGamma(0.1),
                                AdjustSaturation(0.1),
                                AdjustContrast(0.1),
                                #AdjustHue(0.1),
                                AdjustBrightness(0.1)])
            '''
            aug_orignal_txt = Compose([AdjustGamma(0.01)])
                                
            aug_orignal_scl = Compose([
                                RandomRSized(RESIZE, SCALE_SZ2),
                                CenterCrop(RCROP)])
            ##scale samples
            og_img_scl, og_tran_dict = aug_orignal_scl(Image.fromarray(results['img']))
            lbboxes = results['boxes'] * 1
            if 'RandomSized' in og_tran_dict.keys():
                 lbboxes *= og_tran_dict['RandomSized'][0]
                 #print(og_tran_dict['RandomSized'])
                 
            lbboxes[:,0] -= og_tran_dict['CenterCrop'][0]
            lbboxes[:,2] -= og_tran_dict['CenterCrop'][0]
            lbboxes[:,1] -= og_tran_dict['CenterCrop'][1]
            lbboxes[:,3] -= og_tran_dict['CenterCrop'][1]
            
            if results['domain'] == 0:
                mx_vl_0 = lbboxes[:,2].max()
                mx_vl_1 = lbboxes[:,2].min()
                
                mx_vl_3 = lbboxes[:,0].max()
                mx_vl_4 = lbboxes[:,0].min()
                
                my_vl_0 = lbboxes[:,3].max()
                my_vl_1 = lbboxes[:,3].min()
                
                my_vl_3 = lbboxes[:,1].max()
                my_vl_4 = lbboxes[:,1].min()
                
                ledge_x = og_tran_dict['CenterCrop'][4] * 2.8 / 4
                ledge_y = og_tran_dict['CenterCrop'][5] * 2.8 / 4
                if mx_vl_0 <= 0 or mx_vl_4 >= ledge_x or my_vl_0 <= 0 or my_vl_4 >= ledge_y:
                      aug_orignal_scl = Compose([CenterCrop(RCROP)])
                      og_img_scl, og_tran_dict = aug_orignal_scl(Image.fromarray(results['img']))
                      
                      lbboxes = results['boxes'] * 1
                      if 'RandomSized' in og_tran_dict.keys():
                           lbboxes *= og_tran_dict['RandomSized'][0]
                           #print(og_tran_dict['RandomSized'])
                           
                      lbboxes[:,0] -= og_tran_dict['CenterCrop'][0]
                      lbboxes[:,2] -= og_tran_dict['CenterCrop'][0]
                      lbboxes[:,1] -= og_tran_dict['CenterCrop'][1]
                      lbboxes[:,3] -= og_tran_dict['CenterCrop'][1]
                  
            lbboxes[lbboxes<0] = 0
            x1 = lbboxes[:, 2]
            y1 = lbboxes[:, 3]
            x1[x1>=og_tran_dict['CenterCrop'][4]] = og_tran_dict['CenterCrop'][4] - 1
            y1[y1>=og_tran_dict['CenterCrop'][5]] = og_tran_dict['CenterCrop'][5] - 1
             
            lbboxes[:, 2] = x1
            lbboxes[:, 3] = y1
            results['boxes'] = lbboxes
            
            og_img_txt, _ = aug_orignal_txt(og_img_scl)
            
                 
            results['img'] = og_img_txt
            results['img_sc'] = og_img_scl
        else:
            results['img_sc'] = Image.fromarray(results['img'])
            results['img'] = Image.fromarray(results['img'])
                            
        
        ###heanvy disturbance###
        SCALE_SZ0 = [1.0, 2.0, 3.5]
        if random.random() < 0.25:
            transfer_hd = [AdjustGamma(0.6),
                            AdjustSaturation(0.8),
                            AdjustContrast(0.5),
                            AdjustHue(0.5),
                            AdjustSharpness(0.5),
                            AdjustBrightness(0.6)]
        else:
            transfer_hd = [AdjustGamma(0.1),
                            AdjustSaturation(0.1),
                            AdjustContrast(0.1),
                            AdjustHue(0.1),
                            AdjustSharpness(0.1),
                            AdjustBrightness(0.1)]
                        
        if random.random() < 0.2:
            transfer_hd += [Iaaa_Color()]
        
        #transfer_hd += [RandomLSized(RESIZE, SCALE_SZ0),
        #                CenterCrop(RCROP),
        #                RandomHorizontallyFlip(0.15)]
        
        transfer_hd += [CenterCrop(RCROP),
                        RandomHorizontallyFlip(0.15)]
        
        aug_scale = Compose(transfer_hd)
        
        
        ###light distrubance###
        transfer_ld = transfer_hd[:-2]#transfer_hd[-3:]
        aug_transfer = Compose(transfer_ld)
        #print(transfer_ld)
        
                                  
        ##scale samples
        sc_img, sc_tran_dict= aug_scale(results['img_sc'])
        results['sc_tran_dict'] = sc_tran_dict
        
        ##transfer sample
        tr_img, tr_tran_dict= aug_transfer(results['img_sc'])
        results['tr_tran_dict'] = tr_tran_dict
        
        '''
        results['img'].save('./test_0.jpg')
        sc_img.save('./test/'+str(results['img_id']) +'.jpg')
        tr_img.save('./test_2.jpg')
        '''
        
        img = np.array(results['img'], dtype=np.float32)
        img = self.norm_f(img)
        results['img'] = img
        
        
        sc_img = np.array(sc_img, dtype=np.float32)
        sc_img = self.norm_f(sc_img)
        results['sc_img'] = sc_img
        
        tr_img = np.array(tr_img, dtype=np.float32)
        #print('--tr_img:', tr_img.min(), tr_img.max())
        tr_img = self.norm_f(tr_img)
        results['tr_img'] = tr_img

        size_w, size_h = results['img'].shape[1], results['img'].shape[0]
        mask_ = self.make_mask(results['labels'], results['boxes'], size_w, size_h, class_num=8)
        results['gen_masks'] = mask_
        
        return results
    
    def norm_f(self, img):
        if self.to_01:
            img = img / 255.0
        if self.to_bgr:
            img = img[:, :, [2, 1, 0]]
        img = (img - self.mean) / self.std
        
        return img

    def make_mask(self, cls, box, size_w, size_h, class_num=8):
        mask_set = []
        for idx in range(class_num):
            mask_set.append(Image.new('L', (size_w, size_h), 0))

        box_ = np.round(box / 4) * 4
        box_ = box_.astype('int')
        w = box_[:,2] - box_[:,0]
        h = box_[:,3] - box_[:,1]

        for idx in range(cls.size):
            idx_cls = cls[idx]
            if idx_cls>class_num:
                continue
            sub = Image.new('L', (w[idx], h[idx]), 255)
            mask_set[idx_cls-1].paste(sub, (box_[idx, 0], box_[idx, 1]))
                
        
        for idx in range(class_num):
            mask_set[idx] = np.array(mask_set[idx], dtype=np.float32)
            
        return mask_set


class pad(object):
    def __init__(self, size_divisor=0, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        img = results['img']
        pad_shape = img.shape
        if self.size_divisor > 0:
            pad_shape = list(pad_shape)
            pad_shape[0] = int(np.ceil(img.shape[0] / self.size_divisor)) * self.size_divisor
            pad_shape[1] = int(np.ceil(img.shape[1] / self.size_divisor)) * self.size_divisor
            pad_shape = tuple(pad_shape)
            pad = np.full(pad_shape, self.pad_val, dtype=img.dtype)
            pad[:img.shape[0], :img.shape[1], ...] = img
        else:
            pad = img
        results['img'] = pad
        results['pad_shape'] = pad_shape
        return results


def de_normalize(image, img_meta):
    assert 'img_norm' in img_meta
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = image * img_meta['img_norm']['std'] + img_meta['img_norm']['mean']
    if img_meta['img_norm']['to_01']:
        image *= 255
    if img_meta['img_norm']['to_bgr']:
        image = image[:, :, [2, 1, 0]]
    image = image.astype('uint8')
    return image


class collect(object):

    def __init__(self, meta_keys=('img_shape', 'flip', 'origin_img_shape', 'scale_factor', 'img_norm', 'img_info', 'pad_shape')):
        self.meta_keys = meta_keys

    def __call__(self, results):
        
        img = torch.from_numpy(results['img'].transpose(2, 0, 1))
        sc_img = torch.from_numpy(results['sc_img'].transpose(2, 0, 1))
        sc_tran_dict = results['sc_tran_dict']
        tr_img = torch.from_numpy(results['tr_img'].transpose(2, 0, 1))
        tr_tran_dict = results['tr_tran_dict']
        
        target = {
            'boxes': torch.from_numpy(results['boxes'].astype(np.float32)),
            'labels': torch.from_numpy(results['labels']),
        }
        
        if 'masks' in results:
            target['masks'] = PolygonMasks(results['masks'])
        
        ngen_masks = []
        for idx in range(len(results['gen_masks'])):
            ngen_masks.append(torch.from_numpy(results['gen_masks'][idx]))
        ngen_masks = torch.stack(ngen_masks, 0)
        s_mask = torch.sum(ngen_masks, 0)
        m = torch.where(s_mask == 0, torch.ones(1, ngen_masks.size(-2), ngen_masks.size(-1)),
                        torch.zeros(1,  ngen_masks.size(-2), ngen_masks.size(-1)))
        ngen_masks = torch.cat((m, ngen_masks), 0)

        img_meta = {
        }
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        target = Container(target)
        return img, img_meta, target, sc_img, sc_tran_dict, tr_img, tr_tran_dict, ngen_masks


class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
        return results
