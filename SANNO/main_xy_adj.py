from config import parser_add_main_args
from utils import seed_everything, MemoryQueue, entropy
from datasets_adj import load_data
import datetime
from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from easydl import inverseDecaySheduler, OptimWithSheduler, TrainingModeManager, OptimizerManager, AccuracyCounter
from easydl import one_hot, variable_to_numpy, clear_output
from model import MLP, CLS, ProtoCLS, MLPTrans, MLPTransXY, GTransXY
from tqdm import tqdm
from lib_ent import sinkhorn, ubot_CCD, adaptive_filling, entropy_loss
import ot
import pandas as pd
import numpy as np
from eval_ent_mha_xy import eval, get_embedding
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from utils import indices_values_to_sparse_mx, adj_mul, adj_storage


def main():
    # get args
    args = parser_add_main_args()
    
    # set seed
    seed_everything(args.seed)
    if args.type =="sc2sc":
        print("Use singlecell to annote singlecell")
    elif args.type =="st2st":
        print("Use Spatially Resolved Single-Cell Data to annote Spatially Resolved scdata")
    else:
        print("Use singlecell data to annote st data")

    classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl, graph, inverse_dict = load_data(args)
    print("Load Data Done!")
    # set log
    log_path = args.log
    log_path = f"{log_path}/<dataset>/<now>_<name>"
    log_path = log_path.replace("<dataset>", args.dataset)
    now = datetime.datetime.now().strftime('%b%d_%H-%M')
    log_path = log_path.replace("<now>", now)
    log_path = log_path.replace("<name>", args.name)

    log_dir = f'{log_path}'
    logger = SummaryWriter(log_dir)

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")

    # define network architecture
    cls_output_dim = len(classes_set['source_classes'])
    feature_extractor = GTransXY(in_dim, args.hidden_dim, num_layers=args.num_layers)
    # feature_extractor = MLPTransXY(in_dim, args.hidden_dim)
    classifier = CLS(feature_extractor.output_dim, cls_output_dim, hidden_mlp=args.hidden_dim, feat_dim=args.feat_dim, temp=args.temp)
    cluster_head = ProtoCLS(args.feat_dim, args.K, temp=args.temp)

    feature_extractor = feature_extractor.to(device)
    classifier = classifier.to(device)
    cluster_head = cluster_head.to(device)
    print('feature_extractor : ', feature_extractor)
    optimizer_featex = optim.SGD(feature_extractor.parameters(), lr=args.lr*0.1, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)
    # optimizer_featex = optim.Adam(feature_extractor.parameters(),weight_decay=args.weight_decay, lr=args.lr*0.1)
    optimizer_cls = optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)
    optimizer_cluhead = optim.SGD(cluster_head.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)

    # learning rate decay
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=args.min_step)
    opt_sche_featex = OptimWithSheduler(optimizer_featex, scheduler)
    opt_sche_cls = OptimWithSheduler(optimizer_cls, scheduler)
    opt_sche_cluhead = OptimWithSheduler(optimizer_cluhead, scheduler)

    # adj_spatial_source = graph['spatial_source']
    # adj_spatial_source = adj_spatial_source.to(device)
    # adj_spatial_target = graph['spatial_target']
    # adj_spatial_target = adj_spatial_target.to(device)
    # adj_feature_source = graph['feature_source']
    # adj_feature_source = adj_feature_source.to(device)
    # adj_feature_target = graph['feature_target']
    # adj_feature_target = adj_feature_target.to(device)
    adj_matrix_source = graph['source']
    adj_matrix_source = adj_matrix_source.to(device)
    adj_matrix_target = graph['target']
    adj_matrix_target = adj_matrix_target.to(device)

    adjs_source = adj_storage(adj_matrix_source.coalesce().indices(), graph['source'].size(0))
    adjs_target = adj_storage(adj_matrix_target.coalesce().indices(), graph['target'].size(0))
    n_target=graph['target'].size(0)
    n_source = graph['source'].size(0)
    # Memory queue init
    n_batch = int(args.MQ_size/args.batch_size)    
    memqueue = MemoryQueue(args.feat_dim, args.batch_size, n_batch, device, args.temp).to(device)
    cnt_i = 0
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
        while cnt_i < n_batch:
            for i, (im_target, _, x_target, y_target, id_target) in enumerate(target_initMQ_dl):
                im_target = im_target.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)
                id_target = id_target.to(device)
                
                t_sp_adjs_i = []
                t_sp_edge_index_i, _ = subgraph(id_target, adjs_target[0], num_nodes=n_target, relabel_nodes=True)
                t_sp_adjs_i.append(t_sp_edge_index_i.to(device))
                for k in range(1):
                    t_sp_edge_index_i, _ = subgraph(id_target, adjs_target[k+1], num_nodes=n_target, relabel_nodes=True)
                    t_sp_adjs_i.append(t_sp_edge_index_i.to(device))
                
                feature_ex, _ = feature_extractor(im_target, t_sp_adjs_i)
                before_lincls_feat, after_lincls = classifier(feature_ex)

                memqueue.update_queue(F.normalize(before_lincls_feat), id_target)
                cnt_i += 1
                if cnt_i > n_batch-1:
                    break

    total_steps = tqdm(range(args.min_step), desc='global step')
    global_step = 0
    beta = None

    while global_step < args.min_step:
        iters = zip(source_train_dl, target_train_dl)
        for minibatch_id, ((im_source, label_source, x_source, y_source, id_source), (im_target, _, x_target, y_target, id_target)) in enumerate(iters):
            label_source = label_source.to(device)
            im_source = im_source.to(device)
            x_source = x_source.to(device)
            y_source = y_source.to(device)
            id_source = id_source.to(device)
            s_sp_adjs_i = []
            # import pdb;pdb.set_trace()
            s_sp_edge_index_i, _ = subgraph(id_source, adjs_source[0], num_nodes=n_source, relabel_nodes=True)
            s_sp_adjs_i.append(s_sp_edge_index_i.to(device))
            for k in range(1):
                s_sp_edge_index_i, _ = subgraph(id_source, adjs_source[k+1], num_nodes=n_source, relabel_nodes=True)
                s_sp_adjs_i.append(s_sp_edge_index_i.to(device))

            im_target = im_target.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)
            id_target = id_target.to(device)

            t_sp_adjs_i = []
            t_sp_edge_index_i, _ = subgraph(id_target, adjs_target[0], num_nodes=n_target, relabel_nodes=True)
            t_sp_adjs_i.append(t_sp_edge_index_i.to(device))
            for k in range(1):
                t_sp_edge_index_i, _ = subgraph(id_target, adjs_target[k+1], num_nodes=n_target, relabel_nodes=True)
                t_sp_adjs_i.append(t_sp_edge_index_i.to(device))


            feature_ex_s,loss_fs = feature_extractor.forward(im_source, s_sp_adjs_i)
            feature_ex_t,loss_ft = feature_extractor.forward(im_target, t_sp_adjs_i)


            before_lincls_feat_s, after_lincls_s = classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)
        
            norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            after_cluhead_t = cluster_head(before_lincls_feat_t)

            # =====Source Supervision=====
            criterion = nn.CrossEntropyLoss().to(device)
            loss_cls = criterion(after_lincls_s, label_source)

            # =====Private Class Discovery=====
            minibatch_size = norm_feat_t.size(0)

            # obtain nearest neighbor from memory queue and current mini-batch
            feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / args.temp
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(device)
            feat_mat2.masked_fill_(mask, -1/args.temp)

            nb_value_tt, nb_feat_tt = memqueue.get_nearest_neighbor(norm_feat_t, id_target.to(device))
            neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1,1), feat_mat2], 1)
            values, indices = torch.max(neighbor_candidate_sim, 1)
            
            neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).to(device)
            for i in range(minibatch_size):
                neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1,-1), norm_feat_t], 0)
                neighbor_norm_feat[i,:] = neighbor_candidate_feat[indices[i],:]

            neighbor_output = cluster_head(neighbor_norm_feat)
            
            # # fill input features with memory queue
            # fill_size_ot = args.K
            # mqfill_feat_t = memqueue.random_sample(fill_size_ot)
            # mqfill_output_t = cluster_head(mqfill_feat_t)

            # OT process
            # mini-batch feat (anchor) | neighbor feat | filled feat (sampled from memory queue)
            # S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
            S_tt = torch.cat([after_cluhead_t, neighbor_output], 0)
            S_tt *= args.temp
            Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
            Q_tt_tilde = Q_tt * Q_tt.size(0)
            anchor_Q = Q_tt_tilde[:minibatch_size, :]
            neighbor_Q = Q_tt_tilde[minibatch_size:2*minibatch_size, :]

            # compute loss_PCD
            loss_local = 0
            for i in range(minibatch_size):
                sub_loss_local = 0
                sub_loss_local += -torch.sum(neighbor_Q[i,:] * F.log_softmax(after_cluhead_t[i,:]))
                sub_loss_local += -torch.sum(anchor_Q[i,:] * F.log_softmax(neighbor_output[i,:]))
                sub_loss_local /= 2
                loss_local += sub_loss_local
            loss_local /= minibatch_size
            loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
            # loss_PCD = args.lam_local * loss_local + args.lam_global * loss_global
            loss_PCD = (loss_global + loss_local) / 2

            # =====Common Class Detection=====
            if global_step > 500:
                source_prototype = classifier.ProtoCLS.fc.weight
                if beta is None:
                    beta = ot.unif(source_prototype.size()[0])

                # fill input features with memory queue
                fill_size_uot = n_batch*args.batch_size
                mqfill_feat_t = memqueue.random_sample(fill_size_uot)
                ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
                full_size = ubot_feature_t.size(0)
                
                # Adaptive filling
                newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, args.gamma, beta, fill_size_uot, device)

                # UOT-based CCD
                high_conf_label_id, high_conf_label, _, new_beta, k_weight, u_weight = ubot_CCD(newsim, beta, fake_size=fake_size, device=device,
                                                                        fill_size=fill_size_uot, mode='minibatch')
                # adaptive update for marginal probability vector
                beta = args.mu*beta + (1-args.mu)*new_beta

                if high_conf_label_id.size(0) > 0:
                    loss_CCD = criterion(after_lincls_t[high_conf_label_id,:], high_conf_label[high_conf_label_id])
                else:
                    loss_CCD = 0
                t_prediction = F.softmax(after_lincls_t, dim=1)
                loss_ent = - args.lam_pe * entropy_loss(t_prediction, k_weight) - args.lam_ne * entropy_loss(t_prediction, u_weight)
            else:
                loss_CCD = 0
                loss_ent = 0

            loss_all = 2 * loss_cls + args.lam_PCD * loss_PCD + args.lam_CCD * loss_CCD + args.lam_ent * loss_ent - args.lam_link * sum(loss_fs) / len(loss_fs)  - args.lam_link * sum(loss_ft) / len(loss_ft)

            with OptimizerManager([opt_sche_featex, opt_sche_cls, opt_sche_cluhead]):
                loss_all.backward()

            classifier.ProtoCLS.weight_norm() # very important for proto-classifier
            cluster_head.weight_norm() # very important for proto-classifier
            memqueue.update_queue(norm_feat_t, id_target.to(device))
            global_step += 1
            total_steps.update()

            if global_step % args.log_interval == 0:
                counter = AccuracyCounter()
                counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(classes_set['source_classes']))), variable_to_numpy(after_lincls_s))
                acc_source = torch.tensor([counter.reportAccuracy()]).to(device)
                logger.add_scalar('loss_all', loss_all, global_step)
                logger.add_scalar('loss_cls', loss_cls, global_step)
                logger.add_scalar('loss_PCD', loss_PCD, global_step)
                logger.add_scalar('loss_CCD', loss_CCD, global_step)
                logger.add_scalar('acc_source', acc_source, global_step)

            if global_step > 500 and global_step % args.test_interval == 0:
                results = eval(feature_extractor, classifier, target_test_dl, classes_set, adjs_target, n_target, device, gamma=args.gamma, beta=beta)

                df = pd.DataFrame(results['report_1']).T
                df.to_csv(f"{log_dir}/report_1_{global_step}.csv")

                df = pd.DataFrame(results['report_2']).T
                df.to_csv(f"{log_dir}/report_2_{global_step}.csv")

                df = pd.DataFrame(results['report_3']).T
                df.to_csv(f"{log_dir}/report_3_{global_step}.csv")

                df = pd.DataFrame(results['report_4']).T
                df.to_csv(f"{log_dir}/report_4_{global_step}.csv")

                logger.add_scalar('cls_common_acc', results['cls_common_acc'], global_step)
                logger.add_scalar('cls_tp_acc', results['cls_tp_acc'], global_step)
                logger.add_scalar('cls_overall_acc', results['cls_overall_acc'], global_step)

                # s_feat, t_feat, s_gt, t_gt = get_embedding(feature_extractor, classifier, source_test_dl, target_test_dl,adjs_sp_target, adjs_feat_target, n_target, adjs_sp_source, adjs_feat_source, n_source, device)
                # import scanpy as sc
                # adata = sc.AnnData(np.concatenate((s_feat, t_feat), axis=0))
                # adata.obs["CellType"] = np.concatenate((s_gt, t_gt), axis=0)
                # adata.obs["Batch"] = ["training set"] * s_feat.shape[0] + ["test set"] * t_feat.shape[0]
                # adata.write(f'{log_dir}/embedding_{global_step}.h5ad')

    results = eval(feature_extractor, classifier, target_test_dl, classes_set, adjs_target, n_target, device, gamma=args.gamma, beta=beta)
    s_feat, t_feat, s_gt, t_gt = get_embedding(feature_extractor, classifier, source_test_dl, target_test_dl,adjs_target, n_target, adjs_source, n_source, device)
    import scanpy as sc
    adata = sc.AnnData(t_feat)
    adata.obs["CellType"] = t_gt


    pred_df = results['report_4']
    # uniformed_index = len(classes_set['source_classes'])
    # inverse_dict[uniformed_index] = "novel"

    ## I think is need to be modified!!!
    inverse_dict.update({cls: "novel" for cls in classes_set['tp_classes']})


    pred_df['pred_str'] = [inverse_dict[i] for i in pred_df['pred_ot']]

    adata.obs['pred'] = pred_df['pred_ot'].values
    adata.obs['pred_str'] = pred_df['pred_str'].values

    adata.write(f'{log_dir}/embedding.h5ad')

    # adata = sc.AnnData(np.concatenate((s_feat, t_feat), axis=0))
    # adata.obs["CellType"] = np.concatenate((s_gt, t_gt), axis=0)
    # adata.obs["Batch"] = ["training set"] * s_feat.shape[0] + ["test set"] * t_feat.shape[0]
    # adata.write(f'{log_dir}/embedding.h5ad')

    acc_df = pd.DataFrame({
        "seen_acc": results['common_acc'],
        "novel_acc": results['novel_acc'],
        "overall_acc": results['overall_acc'],
    }, index=[0])

    acc_df.to_csv(f"{log_dir}/acc.csv")

if __name__ == '__main__':
    main()