import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import random
#from kmeans_pytorch import kmeans
from .SampleMemory import SamplerMemory
from .SampleAugmentation import SamplerAugmentation

class MemoryManagement(nn.Module):
    def __init__(self, num_classes):
        super(MemoryManagement, self).__init__()
        #label params
        self.class_numbers = num_classes
        #self.n_cluster = 10
        
        
        self.source_memory = SamplerMemory(self.class_numbers, [1280, 720], domain='s')
        #self.source_aug = SamplerAugmentation(self.class_numbers, [1280, 720], is_translate=True)
        
        
        self.target_memory = SamplerMemory(self.class_numbers, [1024, 512], domain='t')
        #self.target_aug = SamplerAugmentation(self.class_numbers, [1024, 512])
        
        
    
    def forward(self, images=None, targets=None, img_metas=None, method='r', device=None, domain='s', read_sample_index=None):
        if method == 'r':
             if domain == 's':
                 limages, ltargets, lmetas = self.source_memory(device, read_sample_index=read_sample_index)
                 #return {'images': limages, 'targets': ltargets, 'img_metas': lmetas}
                 return limages, ltargets, lmetas
             elif domain == 't':
                 limages, ltargets, lmetas = self.target_memory(device, read_sample_index=read_sample_index)
                 #return {'images': limages, 'targets': ltargets, 'img_metas': lmetas}
                 return limages, ltargets, lmetas
        elif method == 'w':
             if domain == 's':
                 #images, targets = self.source_aug(images, targets)
                 self.source_memory(images, targets, img_metas, method='w')
             elif domain == 't':
                 #images, targets = self.source_aug(images, targets)
                 self.target_memory(images, targets, img_metas, method='w')
        
    
    def set_status_update(self, status_update, domain='s'):
        if domain == 's':
             print('----:', domain)
             self.source_memory.set_status_update(status_update=status_update)
        elif domain == 't':
             print('----:', domain)
             self.target_memory.set_status_update(status_update=status_update)
    
    def clear_fucn(self, domain='s'):
        if domain == 's':
             self.source_memory.clear()
        elif domain == 't':
             self.target_memory.clear()
    

    def set_samples(self, domain='s'):
        if domain == 's':
             return self.source_memory.set_samples()
        elif domain == 't':
             return self.target_memory.set_samples()


