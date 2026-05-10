import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import random
#from kmeans_pytorch import kmeans

class SamplerMemory(nn.Module):
    def __init__(self, num_classes, im_szs, domain='s'):
        super(SamplerMemory, self).__init__()
        #label params
        self.class_numbers = num_classes
        
        #memory
        self.memory_total_sz = 70#80#70
        self.images_sample_memory = [[None for i in range(self.memory_total_sz)] for idx in range(self.class_numbers)]
        self.targets_sample_memory = [[None for i in range(self.memory_total_sz)] for idx in range(self.class_numbers)]
        self.metas_sample_memory = [[None for i in range(self.memory_total_sz)] for idx in range(self.class_numbers)]
        
        self.sample_memory_ids = np.zeros((self.class_numbers, self.memory_total_sz)) - 1
        self.sample_memory_loc = np.zeros((self.class_numbers))
        self.sample_memory_ovf = np.zeros((self.class_numbers))
        self.sample_memory_mark = np.zeros((self.class_numbers, self.memory_total_sz))
        
        #self.update_cluster_center = True
        self.state_memory = False
        
        self.read_sample_index = 0

        self.init_read_mark = False
        self.sample_index_list = []
        self.shuffle_index_list = []


        self.sample_cls_batch = 32#60#8#50#16
        self.sample_batch = 16

        self.status_update = False
        self.mem_domain = domain
    
    def clear(self):
        self.sample_memory_index *= 0
        self.sample_memory_loc *= 0
        self.sample_memory_ovf *= 0

        self.sample_memory_index_mem *= 0
        self.sample_memory_score_mem *= 0
        
        #self.image_sample_memory *= 0
        #self.mask_sample_memory *= 0
        
        self.state_memory = False
        self.status_update = False
        
    def write_memory(self, images, targets, img_metas, status_update=False):
       bn, dm, fh, fw = images.size()
       sample_memory_ovf = (self.sample_memory_ovf- self.sample_memory_ovf.min()) * self.memory_total_sz + self.sample_memory_loc
       sort_index = np.argsort(sample_memory_ovf)

       #all_label_id_list = np.array(self.sample_memory_ids, dtype=np.int32)
       #all_label_id_list_mask = np.array(all_label_id_list > -0.5, dtype=np.int32)
       #all_label_id_list_mask = all_label_id_list_mask.sum(axis=1)
       #sort_index = np.argsort(all_label_id_list_mask)

       for idx_bn in range(bn):
            ltargets = targets[idx_bn]
            #label_set = np.unique(label)
            label_set = ltargets['labels'].cpu().detach().numpy()
            limg_metas = img_metas[idx_bn]
            
            for idx in range(len(sort_index)):
                label_v = sort_index[idx]
               
                if (label_v+1) in label_set and ltargets['boxes'].size(0) > 0:
                    image_id = limg_metas['img_info']['id']
                    label_id_list = np.array(self.sample_memory_ids[sort_index[0:idx+1]], dtype=np.int32)
                    mask_is_in = np.array(label_id_list == image_id)
                    if mask_is_in.max() > 0.5:  #存在
                        break
                    else:                       #不存在
                        #获取序号
                        cur_label_id_list = self.sample_memory_ids[label_v]
                        mask_blank_loc = np.array(cur_label_id_list < -0.5, dtype=np.int32)
                        if mask_blank_loc.max() > 0.5:
                            label_indexes = np.argsort(cur_label_id_list)
                            sample_memory_loc = label_indexes[0]
                        else:
                            #sample_index = random.sample(range(0,len(cur_label_id_list)),1)
                            #sample_memory_loc = sample_index[0]
                            
                            sample_memory_mark = self.sample_memory_mark[label_v]
                            sample_memory_mark_index = np.argsort(sample_memory_mark)
                            sample_memory_loc = sample_memory_mark_index[0]
                            
                            #sample_memory_loc = int(self.sample_memory_loc[label_v])
                        
                        #写入
                        self.images_sample_memory[label_v][sample_memory_loc] = images[idx_bn].cpu()
                        nltargets = {}
                        for key, value in ltargets.items():
                            if key in ['masks', 'gl_scores']:
                                nltargets[key] = value
                            else:
                                nltargets[key] = value.cpu()
                        
                        self.targets_sample_memory[label_v][sample_memory_loc] = nltargets
                        self.metas_sample_memory[label_v][sample_memory_loc] = img_metas[idx_bn]
                        self.sample_memory_ids[label_v, sample_memory_loc] = image_id
                        
                        mask_label_ = np.array(label_set == (label_v+1), dtype=np.int32)
                        mask_label_ = mask_label_.sum()
                        self.sample_memory_mark[label_v, sample_memory_loc] = mask_label_

                        #消除后面其他重复部分
                        all_label_id_list = np.array(self.sample_memory_ids, dtype=np.int32)
                        overlap_mask = np.array(all_label_id_list == image_id, dtype=np.int32)
                        overlap_mask[sort_index[0:idx+1]] = 0
                        self.sample_memory_ids = self.sample_memory_ids * (1-overlap_mask) + (-1) * overlap_mask

                        #更新操作次数
                        self.sample_memory_loc[label_v] += 1
                        if self.sample_memory_loc[label_v] == self.memory_total_sz:
                            self.sample_memory_loc[label_v] = 0
                            self.sample_memory_ovf[label_v] += 1
                        
                    break
                
                         

           
       return
    
    def read_memory(self, device, status_update=False, read_sample_index=None):
        if len(self.sample_index_list) > 0:
            if read_sample_index is not None:
                self.read_sample_index = read_sample_index

            ims = []
            targets = []
            img_metas = []
            for idx_k in range(self.sample_batch):
                lindex = (read_sample_index * self.sample_batch + idx_k) % len(self.sample_index_list)
                mem_type, cls_idx, sample_idx = self.sample_index_list[lindex]
                
                if mem_type == 0:
                    im = self.images_sample_memory[cls_idx][sample_idx]
                    target = self.targets_sample_memory[cls_idx][sample_idx]
                    img_meta = self.metas_sample_memory[cls_idx][sample_idx]

                nmtgt_targets = {}
                for key, value in target.items():
                    if key in ['masks', 'gl_scores']:
                        nmtgt_targets[key] = value
                    else:
                        nmtgt_targets[key] = value.to(device)
                
                ims.append(im.to(device))
                targets.append(nmtgt_targets)
                img_metas.append(img_meta)
                
            ims = torch.stack(ims, dim=0)
            return ims, targets, img_metas
        else:
            return None, None, None
            
        
    def forward(self, images=None, targets=None, img_metas=None, method='r', device=None, read_sample_index=None):
        if method == 'r':
             return self.read_memory(device, read_sample_index=read_sample_index)
        elif method == 'w':
             self.write_memory(images, targets, img_metas)
             return
    
    def set_status_update(self, status_update):
        self.status_update = status_update
    
    def set_samples(self):
        sample_memory_ids_all = self.sample_memory_ids
        sample_memory_ids_all_mask = np.array(sample_memory_ids_all > -0.5)
        sample_memory_ids_all_mask = sample_memory_ids_all_mask.sum(axis=1)
        print(sample_memory_ids_all_mask.min())
        if sample_memory_ids_all_mask.min() > self.sample_cls_batch / 2:
            if self.status_update:
                sample_index_list = []
                for idx_i in range(self.class_numbers):
                    cls_index = idx_i 
                    sample_memory_ids = np.array(self.sample_memory_ids[cls_index],dtype=np.int32)
                    sample_memory_ids_index = np.arange(0, len(sample_memory_ids), 1)
                    sample_memory_mark = self.sample_memory_mark[cls_index]
                    mask_sms = sample_memory_ids > -0.5
                    sample_memory_index_list = sample_memory_ids_index[mask_sms]
                    sample_memory_mark_list = sample_memory_mark[mask_sms]
                    sample_memory_index = len(sample_memory_index_list)

                    if sample_memory_index > self.sample_cls_batch:
                        #sample_index = random.sample(range(0,sample_memory_index),self.sample_cls_batch)
                        sample_memory_mark_list_index = np.argsort(sample_memory_mark_list)
                        for ll in range(self.sample_cls_batch):
                            ll_setp = 1 * ll
                            lsample_index = sample_memory_mark_list_index[sample_memory_index - ll_setp - 1]
                            sample_index_list.append([0, cls_index, sample_memory_index_list[lsample_index]])
                    else:
                        sample_index = random.sample(range(0,sample_memory_index),sample_memory_index)
                        for ll in range(len(sample_index)):
                            sample_index_list.append([0, cls_index, sample_memory_index_list[sample_index[ll]]])
                
                random.shuffle(sample_index_list)
                self.sample_index_list = sample_index_list
                
                self.read_sample_index = 0
                self.status_update = False
        
        if (len(self.sample_index_list) % self.sample_batch) == 0:
            iter_sample_total = int(len(self.sample_index_list) / self.sample_batch)
        else:
            iter_sample_total = int(len(self.sample_index_list) / self.sample_batch) + 1
        
        return iter_sample_total
        
    



