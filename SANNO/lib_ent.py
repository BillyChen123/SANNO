import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import ot

def sinkhorn(out, epsilon, sinkhorn_iterations):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def ubot_CCD(sim, beta, fake_size=0, fill_size=0, device=None, mode='minibatch', stopThr=1e-4):
    # fake_size (Adaptive filling) + fill_size (memory queue filling) + mini-batch size
    M = -sim                         
    alpha = ot.unif(sim.size(0))
    
    Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(alpha, beta, M.detach().cpu().numpy(), 
                                                    reg=0.01, reg_m=0.6, stopThr=stopThr) 
    Q_st = torch.from_numpy(Q_st).float().to(device)

    # make sum equals to 1
    sum_pi = torch.sum(Q_st)
    Q_st_bar = Q_st/sum_pi
    
    # highly confident target samples selected by statistics mean
    if mode == 'minibatch':
        Q_anchor = Q_st_bar[fake_size+fill_size:, :]
        m = sim[fake_size+fill_size:, :]
    if mode == 'all':
        Q_anchor = Q_st_bar
        m = sim

    k = round(0.25 * Q_anchor.size(0))
    min_dist, _ = torch.min(m, dim=1)
    _, indicate = min_dist.topk(k=k, dim=0)

    k_weight = torch.tensor([Q_anchor[i, :].sum() for i in range(Q_anchor.size(0))]).to(device)
    u_weight = torch.zeros(Q_anchor.size(0)).to(device)
    u_weight[indicate] = 1 - k_weight[indicate]

    # confidence score w^t_i
    wt_i, pseudo_label = torch.max(Q_anchor, 1)
    # confidence score w^s_j
    ws_j = torch.sum(Q_st_bar, 0)

    # filter by statistics mean
    uniformed_index = Q_st_bar.size(1)
    conf_label = torch.where(wt_i > 1/Q_st_bar.size(0), pseudo_label, uniformed_index)
    high_conf_label = conf_label.clone()
    source_private_label = torch.nonzero(ws_j < 1/Q_st_bar.size(1))
    for i in source_private_label:
        high_conf_label = torch.where(high_conf_label == i, uniformed_index, high_conf_label)
    high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)

    
    # for adaptive update
    new_beta = torch.sum(Q_st_bar,0).cpu().numpy()

    return high_conf_label_id, high_conf_label, conf_label, new_beta, k_weight, u_weight

def adaptive_filling(ubot_feature_t, source_prototype, gamma, beta, fill_size, device, stopThr=1e-4):
    sim = torch.matmul(ubot_feature_t, source_prototype.t())
    max_sim, _ = torch.max(sim,1)
    pos_id = torch.nonzero(max_sim > gamma).reshape(-1)
    pos_rate = pos_id.size(0)/max_sim.size(0)
    pos_num = pos_id.size(0)
    neg_num = max_sim.size(0) - pos_num

    if pos_rate <= 0.5:
        # positive filling
        fake_size = neg_num - pos_num
        if fake_size > 0:
            # do 1st OT find top confident target samples
            high_conf_label_id, _, _, _, _, _ = ubot_CCD(sim, beta, fake_size=0, fill_size=fill_size, device=device,
                                                    mode='all', stopThr=stopThr)
            if high_conf_label_id.size(0) > 0:
                select_id = torch.randint(0, high_conf_label_id.size(0), (fake_size,)).to(device)
                fill_pos = sim[high_conf_label_id[select_id],:] 
                newsim = torch.cat([fill_pos, sim], 0)
            else:
                fake_size = 0
                newsim = sim
        else:
            newsim = sim
    else:
        # negative filling
        fake_size = pos_num - neg_num
        if fake_size > 0:
            farthest_sproto_id = torch.argmin(sim, 1)
            fake_private = 0.5 * ubot_feature_t + 0.5 * source_prototype.data[farthest_sproto_id,:]
            fake_private = F.normalize(fake_private)
            select_id = torch.randint(0, fake_private.size(0), (fake_size,)).to(device)
            fill_neg = fake_private[select_id,:]
            fake_sim = torch.matmul(fill_neg, source_prototype.t())
            newsim = torch.cat([fake_sim, sim], 0)
        else:
            newsim = sim
    
    return newsim, fake_size

def entropy_loss(prediction: torch.Tensor, weight=torch.zeros(1)):
    if weight.size(0) == 1:
        entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
        entropy = torch.mean(entropy)
    else:
        entropy = torch.sum(-prediction * torch.log(prediction + 1e-8), 1)
        entropy = torch.mean(weight * entropy)
    return entropy

def pot_ent(source_prototype, feature_t, device=None):
    a, b = ot.unif(source_prototype.size(0)), ot.unif(feature_t.size(0))
    m = torch.cdist(source_prototype, feature_t) ** 2
    m_max = m.max().detach()
    m = m / m_max
    pi, log = ot.partial.entropic_partial_wasserstein(a, b, m.detach().cpu().numpy(), reg=0.01,
                                                          stopThr=1e-10, log=True)
    pi = torch.from_numpy(pi).float().to(device)

    batch_size = feature_t.size(0)

    plan = pi * batch_size
    k = round(0.25 * batch_size)
    min_dist, _ = torch.min(m, dim=0)
    _, indicate = min_dist.topk(k=k, dim=0)

    k_weight = torch.tensor([plan[:, i].sum() for i in range(batch_size)]).to(device)
    # k_weight /= alpha
    u_weight = torch.zeros(batch_size).to(device)
    u_weight[indicate] = 1 - k_weight[indicate]

    return k_weight, u_weight
