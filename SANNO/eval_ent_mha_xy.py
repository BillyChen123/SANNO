from easydl import variable_to_numpy
from easydl import TrainingModeManager, Accumulator
import numpy as np
from lib_ent import ubot_CCD, adaptive_filling
from utils import ResultsCalculator
import torch
import torch.nn.functional as F
from tqdm import tqdm
import ot
import faiss
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from torch_geometric.utils import subgraph
def run_kmeans(L2_feat, ncentroids, init_centroids=None, seed=None, gpu=False, min_points_per_centroid=1):
    if seed is None:
        seed = int(os.environ['PYTHONHASHSEED'])
    dim = L2_feat.shape[1]
    kmeans = faiss.Kmeans(d=dim, k=ncentroids, seed=seed, gpu=gpu, niter=20, verbose=False, \
                        nredo=5, min_points_per_centroid=min_points_per_centroid, spherical=True)
    if torch.is_tensor(L2_feat):
        L2_feat = variable_to_numpy(L2_feat)
    kmeans.train(L2_feat, init_centroids=init_centroids)
    _, pred_centroid = kmeans.index.search(L2_feat, 1)
    pred_centroid = np.squeeze(pred_centroid)
    return pred_centroid, kmeans.centroids


def eval(feature_extractor, classifier, eval_dl, classes_set, adjs_target, n_target, device,
        gamma=0.7, beta=None, seed=None, uniformed_index=None):
    if seed is None:
        seed = int(os.environ['PYTHONHASHSEED'])
    if uniformed_index is None:
        uniformed_index = len(classes_set['source_classes'])
    if beta is None:
        beta = ot.unif(source_prototype.size()[0])
    
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
            Accumulator(['label_t', 'norm_feat_t', 'pred_t']) as eval_accumulator, \
            torch.no_grad():
        for i, (im_t, label_t, x_t, y_t, id_t) in enumerate(tqdm(eval_dl, desc='testing')):
            im_t = im_t.to(device)
            label_t = label_t.to(device)
            x_t = x_t.to(device)
            y_t = y_t.to(device)
            id_t = id_t.to(device)

            t_sp_adjs_i = []
            t_sp_edge_index_i, _ = subgraph(id_t, adjs_target[0], num_nodes=n_target, relabel_nodes=True)
            t_sp_adjs_i.append(t_sp_edge_index_i.to(device))
            for k in range(1):
                t_sp_edge_index_i, _ = subgraph(id_t, adjs_target[k+1], num_nodes=n_target, relabel_nodes=True)
                t_sp_adjs_i.append(t_sp_edge_index_i.to(device))

            feature_ex_t,_ = feature_extractor.forward(im_t, t_sp_adjs_i)
            before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)
            norm_feat_t = F.normalize(before_lincls_feat_t)
            pred_t = torch.argmax(after_lincls_t, 1)

            val = dict()
            for name in eval_accumulator.names:
                val[name] = locals()[name].cpu().data.numpy()

            # print(val['label_t'].shape)
            # print(val['norm_feat_t'].shape)
            # print(len(eval_accumulator['label_t']))
            # print(len(eval_accumulator['norm_feat_t']))

            eval_accumulator.updateData(val)  

    for x in eval_accumulator:
        val[x] = eval_accumulator[x] 
    label_t = val['label_t']
    norm_feat_t = val['norm_feat_t']
    pred_t = val['pred_t']
    del val

    unif_label = label_t.copy()
    index_list = []
    for i, x in enumerate(unif_label):
        if x < uniformed_index:
            index_list.append(i)
    report_3 = classification_report(unif_label[index_list], pred_t[index_list], output_dict=True)

    # Unbalanced OT
    source_prototype = classifier.ProtoCLS.fc.weight

    stopThr = 1e-6
    # Adaptive filling 
    newsim, fake_size = adaptive_filling(torch.from_numpy(norm_feat_t).to(device), 
                                        source_prototype, gamma, beta, 0, device, stopThr=stopThr)
    # newsim = torch.matmul(torch.from_numpy(norm_feat_t).to(device), source_prototype.t())
    # fake_size = 0

    # obtain predict label
    _, __, pred_label, ___, _, _ = ubot_CCD(newsim, beta, fake_size=fake_size, fill_size=0, device=device, mode='minibatch', stopThr=stopThr)
    pred_label = pred_label.cpu().data.numpy()

    report_4 = pd.DataFrame({
        "label": unif_label,
        "pred": pred_t,
        "pred_ot": pred_label,
    })

    # obtain private samples
    filter = (lambda x: x in classes_set["tp_classes"])
    private_mask = np.zeros((label_t.size,), dtype=bool) 
    for i in range(label_t.size):
        if filter(label_t[i]):
            private_mask[i] = True
    private_feat = norm_feat_t[private_mask, :]
    private_label = label_t[private_mask]

    # obtain results
    ncentroids = len(classes_set["tp_classes"])
    private_pred, _ = run_kmeans(private_feat, ncentroids, init_centroids=None, seed=seed, gpu=True)
    results = ResultsCalculator(classes_set, label_t, pred_label, private_label, private_pred)
    results_dict = {
        'report_1': results.report_1,
        'report_2': results.report_2,
        'report_3': report_3,
        'report_4': report_4,
        'cls_common_acc': results.common_acc_aver,
        'cls_tp_acc': results.tp_acc,
        # 'tp_nmi': results.tp_nmi,
        'cls_overall_acc': results.overall_acc_aver,
        # 'h_score': results.h_score,
        # 'h3_score': results.h3_score
        'common_acc': results.common_acc,
        'novel_acc': results.tp_acc,
        'overall_acc': results.overall_acc,
    }
    return results_dict


def get_embedding(feature_extractor, classifier, s_dl, t_dl,adjs_target, n_target, adjs_source, n_source,device):
    with TrainingModeManager([feature_extractor], train=False) as mgr, \
            Accumulator(['s_feat', 's_gt']) as eval_accumulator, \
            torch.no_grad():
        for i, (im, label, x, y, s_id) in enumerate(tqdm(s_dl, desc='get source feature')):
            s_gt = label.to(device)
            im = im.to(device)
            x = x.to(device)
            y = y.to(device)
            s_id = s_id.to(device)

            s_sp_adjs_i = []
            # import pdb;pdb.set_trace()
            s_sp_edge_index_i, _ = subgraph(s_id, adjs_source[0], num_nodes=n_source, relabel_nodes=True)
            s_sp_adjs_i.append(s_sp_edge_index_i.to(device))
            for k in range(1):
                s_sp_edge_index_i, _ = subgraph(s_id, adjs_source[k+1], num_nodes=n_source, relabel_nodes=True)
                s_sp_adjs_i.append(s_sp_edge_index_i.to(device))
            feature_ex_s,_ = feature_extractor.forward(im, s_sp_adjs_i)

            before_lincls_feat_s, _ = classifier(feature_ex_s)
            s_feat = F.normalize(before_lincls_feat_s)
            # print(s_feat)
            val = dict()
            for name in eval_accumulator.names:
                val[name] = locals()[name].cpu().data.numpy()    # variable.cpu().data.numpy()
            # print(val['s_feat'].shape)
            # print(val['s_gt'].shape)
            # print(len(eval_accumulator['s_feat']))
            # print(len(eval_accumulator['s_gt']))

            eval_accumulator.updateData(val)  # eval_accumulator[variable].append()
    for x in eval_accumulator:
        val[x] = eval_accumulator[x]  # variable = eval_accumulator[variable]
    s_feat = val['s_feat']
    s_gt = val['s_gt']

    with TrainingModeManager([feature_extractor], train=False) as mgr, \
            Accumulator(['t_feat', 't_gt']) as eval_accumulator, \
            torch.no_grad():
        for i, (im, label, x, y, t_id) in enumerate(tqdm(t_dl, desc='get target feature')):
            t_gt = label.to(device)
            im = im.to(device)
            x = x.to(device)
            y = y.to(device)
            t_id = t_id.to(device)

            t_sp_adjs_i = []
            t_sp_edge_index_i, _ = subgraph(t_id, adjs_target[0], num_nodes=n_target, relabel_nodes=True)
            t_sp_adjs_i.append(t_sp_edge_index_i.to(device))
            for k in range(1):
                t_sp_edge_index_i, _ = subgraph(t_id, adjs_target[k+1], num_nodes=n_target, relabel_nodes=True)
                t_sp_adjs_i.append(t_sp_edge_index_i.to(device))
            feature_ex_t,_ = feature_extractor.forward(im, t_sp_adjs_i)

            before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)
            t_feat = F.normalize(before_lincls_feat_t)
            
            val = dict()
            for name in eval_accumulator.names:
                val[name] = variable_to_numpy(locals()[name])    # variable.cpu().data.numpy()
            eval_accumulator.updateData(val)  # eval_accumulator[variable].append()
    for x in eval_accumulator:
        val[x] = eval_accumulator[x]  # variable = eval_accumulator[variable]    
    t_feat = val['t_feat']
    t_gt = val['t_gt']
    return s_feat, t_feat, s_gt, t_gt