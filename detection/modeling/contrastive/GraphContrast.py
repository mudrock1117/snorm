from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, models
from torchvision.ops import boxes as box_ops

from detection.layers import FrozenBatchNorm2d, smooth_l1_loss

class GraphContrastive(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.d_node_w = 1.0
        self.d_edge_w = 1.0
        self.d_triple_w = 1.0
        self.adp_pool = nn.AdaptiveAvgPool2d((3,3))
        self.norm = nn.BatchNorm2d(in_channels, affine=True)

    def forward(self, features_v0, features_v1, proposals):
        N, dm, fh, fw = features_v0.size()
        #features_v0 = self.norm(features_v0)
        #features_v1 = self.norm(features_v1)
        features_v0 = self.adp_pool(features_v0).view(N, dm, -1)
        features_v1 = self.adp_pool(features_v1).view(N, dm, -1)
        pos_feats, neg_feats = self.select_samples(features_v0, features_v1, proposals)

        g_pos_nodes, g_pos_edges, g_pos_triples = self.getGraphInformation(pos_feats)
        g_neg_nodes, g_neg_edges, g_neg_triples = self.getGraphInformation(neg_feats)

        d_nodes = self.metric_nodes(g_pos_nodes, g_neg_nodes)
        d_edges = self.metric_edges(g_pos_edges, g_neg_edges)
        d_triples = self.metric_triples(g_pos_triples, g_neg_triples)

        loss_graph = self.d_triple_w * d_triples.mean()#self.d_node_w * d_nodes.mean() #+ self.d_edge_w * d_edges.mean() + self.d_triple_w * d_triples.mean()
        #print(d_nodes.mean(), d_edges.mean(), d_triples.mean())
        return loss_graph.mean()
    
    @torch.no_grad()
    def select_samples(self, features_v0, features_v1, proposals):
        ns, dm, glen = features_v0.size()
        feats_v0 = features_v0.mean(dim=2)
        feats_v1 = features_v1.mean(dim=2)

        #compute similarity matrix
        sim_mat = torch.einsum('nd, md->nm', feats_v0, feats_v1)
        #select pos samples
        pos_mask = torch.eye(ns).to(features_v0.device).detach()

        pos_feats = torch.stack((features_v0,features_v1), dim=1)

        #select neg_samples
        pos_sim_mat = sim_mat.mul(pos_mask)
        pos_sim_mat, _ = pos_sim_mat.max(dim=1, keepdim=True)
        neg_mask = sim_mat.lt(pos_sim_mat * 0.9).float()

        min_sim_mat, _ = sim_mat.min(dim=1, keepdim=True)
        neg_sim_mat = min_sim_mat.mul(1 - neg_mask) + sim_mat.mul(neg_mask)
        neg_sim_mat_tkv, neg_sim_mat_tki = neg_sim_mat.topk(50, dim=1)
        neg_sim_mat_tki = neg_sim_mat_tki.detach()
        #print(neg_sim_mat.size(), neg_sim_mat_tki.max(), neg_sim_mat.max())

        takle_samples = 50
        takle_num = ns // takle_samples + 1
        neg_feats_v0 = []
        neg_feats_v1 = []
        for idx in range(takle_num):
            start_index = idx * takle_samples
            end_index = (idx + 1) * takle_samples

            if end_index > ns:
                end_index = ns
            if start_index >= ns:
                break

            lft_v0 = features_v0[start_index:end_index, :, :]
            lft_v1 = features_v1[start_index:end_index, :, :]
            lnsmi = neg_sim_mat_tki[start_index:end_index, :]

            lft_v0 = lft_v0.unsqueeze(dim=1).expand(lft_v0.size(0), ns, dm, glen)
            lft_v1 = lft_v1.unsqueeze(dim=1).expand(lft_v1.size(0), ns, dm, glen)
            lnsmi = lnsmi.view(lnsmi.size(0), lnsmi.size(1), 1, 1).expand(lnsmi.size(0), lnsmi.size(1), dm, glen).detach()

            slft_v0 = torch.gather(lft_v0, 1, lnsmi)
            slft_v1 = torch.gather(lft_v1, 1, lnsmi)
            neg_feats_v0.append(slft_v0)
            neg_feats_v1.append(slft_v1)

        neg_feats_v0 = torch.cat(neg_feats_v0, dim=0)
        neg_feats_v1 = torch.cat(neg_feats_v1, dim=0)
        neg_feats = torch.cat((neg_feats_v0, neg_feats_v1), dim=1)

        return pos_feats, neg_feats

    def getGraphInformation(self, graph_feats):
        ns, ms, dm, glen = graph_feats.size()

        #nodels
        g_nodes = graph_feats.permute(0,1,3,2).contiguous()

        #edges
        media_num = glen // 2
        total_num = glen
        index_value = []
        for i in range(total_num):
            if i != media_num:
                index_value.append(i)
        
        e_nodes_index = torch.tensor(index_value).to(graph_feats.device).long().detach()
        e_nodes = torch.index_select(g_nodes, 2, e_nodes_index)
        root_nodes = g_nodes[:,:,media_num:media_num+1,:]
        g_edges = root_nodes.mul(e_nodes)

        #triples
        root_nodes = root_nodes.mul(root_nodes)
        e_nodes = e_nodes.mul(e_nodes)
        root_nodes = root_nodes.expand(ns, ms, total_num-1, dm)
        g_triples = torch.stack((root_nodes, e_nodes, g_edges), dim=3)

        return g_nodes, g_edges, g_triples

    def metric_nodes(self, g_pos_nodes, g_neg_nodes):
        ns, ms_p, gln_p, dm = g_pos_nodes.size()
        _, ms_n, gln_n, _ = g_neg_nodes.size()
        scale_w = (dm * 1.0) ** -0.5

        #pos similarity matrix
        gpn0 = g_pos_nodes[:, 0, :, :].view(ns, gln_p, dm)
        gpn1 = g_pos_nodes[:, 1:ms_p, :, :].view(ns, (ms_p-1)*gln_p, dm)

        sim_mat_p = torch.einsum('ngd,nkd->ngk', gpn0, gpn1)
        sim_mat_p = sim_mat_p.view(ns, gln_p, (ms_p-1), gln_p)
        #sim_mat = sim_mat.premute(0,1,3,2,4).contiguous()
        smmt_p_maxv, smmt_p_maxi = sim_mat_p.max(dim=3)
        #sm_mask = sim_mat.ge(smmt_maxv).float()
        mask_p = torch.zeros_like(sim_mat_p[:,0, :,0])
        for idx in range(gln_p):
            lmask = (smmt_p_maxi == idx).float()
            lmaskv, _ = lmask.max(dim=1)
            mask_p = mask_p + lmaskv

        mask_p /= gln_p
        smmt_p_maxv = smmt_p_maxv.mean(dim=1)
        pos_smmt = smmt_p_maxv.mul(mask_p)

        #neg similarity matrix
        gnn1 = g_neg_nodes.view(ns, ms_n*gln_n, dm)

        sim_mat_n = torch.einsum('ngd,nkd->ngk', gpn0, gnn1)
        sim_mat_n = sim_mat_n.view(ns, gln_n, ms_n, gln_n)
        smmt_n_maxv, smmt_n_maxi = sim_mat_n.max(dim=3)
        mask_n = torch.zeros_like(sim_mat_n[:,0,:,0])
        for idx in range(gln_p):
            lmask = (smmt_n_maxi == idx).float()
            lmaskv, _ = lmask.max(dim=1)
            mask_n = mask_n + lmaskv

        mask_n /= gln_n
        smmt_n_maxv = smmt_n_maxv.mean(dim=1)
        neg_smmt = smmt_n_maxv.mul(mask_n)

        #compute loss
        #print(pos_smmt)
        sq_smmt = torch.cat((pos_smmt, neg_smmt),dim=1)
        sq_smmt_mx, _ = sq_smmt.max(dim=1, keepdim=True)
        sq_smmt = sq_smmt - sq_smmt_mx
        sAttn = F.softmax(sq_smmt * scale_w, dim=1)
        loss = -torch.log(sAttn[:, 0:pos_smmt.size(1)].sum(dim=1) + 1e-5)
        '''
        pos_smmt = torch.exp(scale_w * pos_smmt)
        neg_smmt = torch.exp(scale_w * neg_smmt[:,0:10])
        pos_m0 = pos_smmt.sum(dim=1)
        neg_m0 = neg_smmt.sum(dim=1)
        '''
        
        #pos_probs = pos_m0.div(pos_m0 + neg_m0)
        #loss = -torch.log(pos_probs + 0.5)
        return loss

    def metric_edges(self, g_pos_edges, g_neg_edges):
        ns, ms_p, gln_p, dm = g_pos_edges.size()
        _, ms_n, gln_n, _ = g_neg_edges.size()
        scale_w = (dm * 1.0) ** -0.5

        #pos similarity matrix
        gpn0 = g_pos_edges[:, 0, :, :].view(ns, gln_p, dm)
        gpn1 = g_pos_edges[:, 1:ms_p, :, :].view(ns, (ms_p-1)*gln_p, dm)

        sim_mat_p = torch.einsum('ngd,nkd->ngk', gpn0, gpn1)
        sim_mat_p = sim_mat_p.view(ns, gln_p, (ms_p-1), gln_p)
        smmt_p_maxv, smmt_p_maxi = sim_mat_p.max(dim=3)
        mask_p = torch.zeros_like(sim_mat_p[:,0,:,0])
        for idx in range(gln_p):
            lmask = (smmt_p_maxi == idx).float()
            lmaskv, _ = lmask.max(dim=1)
            mask_p = mask_p + lmaskv

        mask_p /= gln_p
        smmt_p_maxv = smmt_p_maxv.mean(dim=1)
        pos_smmt = smmt_p_maxv.mul(mask_p)

        #neg similarity matrix
        gnn1 = g_neg_edges.view(ns, ms_n*gln_n, dm)

        sim_mat_n = torch.einsum('ngd,nkd->ngk', gpn0, gnn1)
        sim_mat_n = sim_mat_n.view(ns, gln_n, ms_n, gln_n)
        smmt_n_maxv, smmt_n_maxi = sim_mat_n.max(dim=3)
        mask_n = torch.zeros_like(sim_mat_n[:,0,:,0])
        for idx in range(gln_p):
            lmask = (smmt_n_maxi == idx).float()
            lmaskv, _ = lmask.max(dim=1)
            mask_n = mask_n + lmaskv

        mask_n /= gln_n
        smmt_n_maxv = smmt_n_maxv.mean(dim=1)
        neg_smmt = smmt_n_maxv.mul(mask_n)

        #compute loss
        sq_smmt = torch.cat((pos_smmt, neg_smmt),dim=1)
        sq_smmt_mx, _ = sq_smmt.max(dim=1, keepdim=True)
        sq_smmt = sq_smmt - sq_smmt_mx
        sAttn = F.softmax(sq_smmt * scale_w, dim=1)
        loss = -torch.log(sAttn[:, 0:pos_smmt.size(1)].sum(dim=1) + 1e-5)
        
        return loss

    def metric_triples(self, g_pos_triples, g_neg_triples):
        loss = (g_pos_triples[:, 0, :, :] - g_pos_triples[:, 1, :, :].detach()).abs()
        loss = torch.mean(loss.sum(dim=1))
        '''
        ns, ms_p, gln_p, tpl_p, dm = g_pos_triples.size()
        _, ms_n, gln_n, tpl_n, _ = g_neg_triples.size()
        g_pos_triples = g_pos_triples.view(ns, ms_p, gln_p, -1)
        g_neg_triples = g_neg_triples.view(ns, ms_n, gln_n, -1)
        scale_w = (tpl_p * dm * 1.0) ** -0.5

        #pos similarity matrix
        gpn0 = g_pos_triples[:, 0, :, :].view(ns, gln_p, tpl_p*dm)
        gpn1 = g_pos_triples[:, 1:ms_p, :, :].view(ns, (ms_p-1)*gln_p, tpl_p*dm)
        
        ###m1
        sim_mat_p = torch.einsum('ngd,nkd->ngk', gpn0, gpn1)
        ###m2
        mask_pos = ((gpn0 - gpn1) > 0).float()
        com_gpn = gpn1.mul(mask_pos) + gpn0.mul(1 - mask_pos)
        sim_mat_p = com_gpn.sum(dim=2)
        
        sim_mat_p = sim_mat_p.view(ns, gln_p, (ms_p-1), gln_p)
        smmt_p_maxv, smmt_p_maxi = sim_mat_p.max(dim=3)
        mask_p = torch.zeros_like(sim_mat_p[:,0,:, 0])
        for idx in range(gln_p):
            lmask = (smmt_p_maxi == idx).float()
            lmaskv, _ = lmask.max(dim=1)
            mask_p = mask_p + lmaskv

        mask_p /= gln_p
        smmt_p_maxv = smmt_p_maxv.mean(dim=1)
        pos_smmt = smmt_p_maxv.mul(mask_p)

        #neg similarity matrix
        gnn1 = g_neg_triples.view(ns, ms_n*gln_n, tpl_n*dm)

        sim_mat_n = torch.einsum('ngd,nkd->ngk', gpn0, gnn1)
        sim_mat_n = sim_mat_n.view(ns, gln_p, ms_n, gln_n)
        smmt_n_maxv, smmt_n_maxi = sim_mat_n.max(dim=3)
        mask_n = torch.zeros_like(sim_mat_n[:,0,:,0])
        for idx in range(gln_p):
            lmask = (smmt_n_maxi == idx).float()
            lmaskv, _ = lmask.max(dim=1)
            mask_n = mask_n + lmaskv

        mask_n /= gln_n
        smmt_n_maxv = smmt_n_maxv.mean(dim=1)
        neg_smmt = smmt_n_maxv.mul(mask_n)

        #compute loss
        sq_smmt = torch.cat((pos_smmt, neg_smmt),dim=1)
        #sq_smmt_mx, _ = pos_smmt.min(dim=1, keepdim=True)
        sq_smmt = sq_smmt.div(sq_smmt.mean(dim=1, keepdim=True))
        #sq_smmt = sq_smmt - sq_smmt_mx
        sAttn = F.softmax(sq_smmt * scale_w, dim=1)
        loss = -torch.log(sAttn[:, 0:pos_smmt.size(1)].sum(dim=1) + 1e-5)
        '''
        return loss*100
