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
        
        #source: 421
        '''
        aug_scale = Compose([AdjustGamma(0.5),
                            AdjustSaturation(0.5),
                            AdjustHue(0.5),
                            AdjustContrast(0.5),
                            RandomSized(RESIZE, SCALE_SZ0),
                            CenterCrop(RCROP)])
        '''
        
        #test: v2-9-5
        '''
        aug_scale = Compose([Iaaa_Color(),
                             Iaaa_Cutout(),
                             RandomSized(RESIZE, SCALE_SZ0),
                             CenterCrop(RCROP)])
        
        '''
        SCALE_SZ0 = [1.0, 2.0, 3.5]
        aug_scale = Compose([Iaaa_Color(),
                            #Iaaa_Cutout(),
                            RandomLSized(RESIZE, SCALE_SZ0),
                            CenterCrop(RCROP),
                            RandomHorizontallyFlip(0.15)])
        
        
        #test: v2-6
        '''
        aug_scale = Compose([AdjustGamma(0.5),
                             AdjustSaturation(0.5),
                             AdjustHue(0.5),
                             AdjustContrast(0.5),
                             RandomSized(RESIZE, SCALE_SZ0),
                             CenterCrop(RCROP),
                             IaRandGaussianNoise(p=0.1),
                             RandomHorizontallyFlip(0.15)
                             ])
        '''
        
        
        
        '''
        #0.376
        aug_transfer = Compose([AdjustGamma(aga),
                                AdjustSaturation(asn),
                                AdjustHue(ahe),
                                AdjustContrast(act),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomHorizontallyFlip(rhf),
                                RandomVerticallyFlip(rvf)
                                ])
        
        '''
        '''
        aug_transfer = Compose([IaRandGaussianNoise(p=0.15),
                                AdjustGamma(aga),
                                AdjustSaturation(asn),
                                AdjustHue(ahe),
                                AdjustContrast(act),
                                #IaRandGaussianNoise(p=0.3),
                                #RandGreyscale(p=0.1),
                                #IaRandColor(p=0.1),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomHorizontallyFlip(rhf),
                                RandomVerticallyFlip(rvf)
                                ])
        '''
        
        #v51
        '''
        aug_transfer = Compose([#IaRandGaussianNoise(p=0.15),
                                AdjustGamma(0.5),
                                AdjustSaturation(0.5),
                                AdjustHue(0.5),
                                AdjustContrast(0.5),
                                #IaRandGaussianNoise(p=0.3),
                                #RandGreyscale(p=0.1),
                                #IaRandColor(p=0.1),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomTranslate([1, 1])
                                ])
        '''
        
        
        
        
        #v52
        '''
        aug_transfer = Compose([#IaRandGaussianNoise(p=0.15),
                                AdjustGamma(0.1),
                                AdjustSaturation(0.1),
                                AdjustHue(0.1),
                                AdjustContrast(0.1),
                                #IaRandGaussianNoise(p=0.3),
                                #RandGreyscale(p=0.1),
                                #IaRandColor(p=0.1),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomTranslate([1, 1])
                                ])
        '''
        #v53
        '''
        SCALE_SZ1 = [0.8, 1.2]
        aug_transfer = Compose([#IaRandGaussianNoise(p=0.15),
                                AdjustGamma(0.5),
                                AdjustSaturation(0.5),
                                AdjustHue(0.5),
                                AdjustContrast(0.5),
                                #IaRandGaussianNoise(p=0.3),
                                #RandGreyscale(p=0.1),
                                #IaRandColor(p=0.1),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomTranslate([1, 1])
                                ])
        '''
        
        #v55
        '''
        SCALE_SZ1 = [0.4, 1.6]
        aug_transfer = Compose([#IaRandGaussianNoise(p=0.15),
                                AdjustGamma(0.5),
                                AdjustSaturation(0.5),
                                AdjustHue(0.5),
                                AdjustContrast(0.5),
                                #IaRandGaussianNoise(p=0.3),
                                #RandGreyscale(p=0.1),
                                #IaRandColor(p=0.1),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomTranslate([1, 1])
                                ])
        '''
        
        #v57
        '''
        aug_transfer = Compose([#IaRandGaussianNoise(p=0.15),
                                AdjustGamma(0.5),
                                AdjustSaturation(0.5),
                                AdjustHue(0.5),
                                AdjustContrast(0.5),
                                #IaRandGaussianNoise(p=0.3),
                                #RandGreyscale(p=0.1),
                                #IaRandColor(p=0.1),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomTranslate([4, 4])
                                ])
        '''
        
        #v59
        '''
        aug_transfer = Compose([#IaRandGaussianNoise(p=0.15),
                                AdjustGamma(0.5),
                                AdjustSaturation(0.5),
                                AdjustHue(0.5),
                                AdjustContrast(0.5),
                                #IaRandGaussianNoise(p=0.3),
                                #RandGreyscale(p=0.1),
                                #IaRandColor(p=0.1),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomTranslate([12, 12]),
                                RandomHorizontallyFlip(0.15),
                                RandomVerticallyFlip(0.15)
                                ])
        '''
        
        #test
        #SCALE_SZ1 = [0.8, 1.2]
        SCALE_SZ1 = [1.0, 2.4]
        aug_transfer = Compose([#IaRandGaussianNoise(p=0.15),
                                AdjustGamma(0.5),
                                AdjustSaturation(0.5),
                                AdjustHue(0.5),
                                AdjustContrast(0.5),
                                RandomLSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomTranslate([12, 12]),
                                RandomHorizontallyFlip(0.15),
                                #RandomVerticallyFlip(0.15)
                                ])
        #tralste
        '''
        aug_transfer = Compose([
                        RandomSized(RESIZE, SCALE_SZ1),
                        CenterCrop(RCROP),
                        RandomTranslate([1, 1])
                        ])
        '''
        '''
        aug_transfer = Compose([Iaaa_Color(),
                                Iaaa_Cutout(),
                                RandomSized(RESIZE, SCALE_SZ1),
                                CenterCrop(RCROP),
                                RandomTranslate([8, 8]),
                                #RandomRotate(20),
                                RandomHorizontallyFlip(rhf),
                                RandomVerticallyFlip(rvf)
                                ])
        '''
        
                                  
        ##scale samples
        sc_img, sc_tran_dict= aug_scale(Image.fromarray(results['img']))
        #sc_img, sc_tran_dict= Image.fromarray(results['img']), {}
        results['sc_tran_dict'] = sc_tran_dict
        
        ##transfer sample
        tr_img, tr_tran_dict= aug_transfer(Image.fromarray(results['img']))
        #tr_img, _ = aug_transfer(Image.fromarray(results['img']))
        results['tr_tran_dict'] = tr_tran_dict
        
        
        img = results['img'].astype(np.float32)
        #print('--img:', img.min(), img.max())
        img = self.norm_f(img)
        results['img'] = img
        
        
        sc_img = np.array(sc_img, dtype=np.float32)
        sc_img = self.norm_f(sc_img)
        results['sc_img'] = sc_img
        
        tr_img = np.array(tr_img, dtype=np.float32)
        #print('--tr_img:', tr_img.min(), tr_img.max())
        tr_img = self.norm_f(tr_img)
        results['tr_img'] = tr_img
        
        results['img_norm'] = dict(mean=self.mean, std=self.std, to_01=self.to_01, to_bgr=self.to_bgr)
        
        return results
    
    def norm_f(self, img):
        if self.to_01:
            img = img / 255.0
        if self.to_bgr:
            img = img[:, :, [2, 1, 0]]
        img = (img - self.mean) / self.std
        
        return img


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

        img_meta = {
        }
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        target = Container(target)
        return img, img_meta, target, sc_img, sc_tran_dict, tr_img, tr_tran_dict


class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
        return results
