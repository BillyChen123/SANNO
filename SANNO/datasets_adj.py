import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.decomposition import PCA
# Dataset
# data_path="./data"
data_path="/bigdat2/user/shanggny/project/STOT/data"
class ST_Dataset(Dataset):
    def __init__(self, data, label,  adj_matrix, x=0, y=0, return_id=True):
        self.datas = data
        self.labels = label
        self.x = np.array(x, dtype='float32')
        self.y = np.array(y, dtype='float32')
        # normalize
        scale = max(self.x.max() - self.x.min(), self.y.max() - self.y.min())
        self.x = (self.x - self.x.min()) / scale
        self.y = (self.y - self.y.min()) / scale
        self.adj_matrix = adj_matrix
        self.return_id = return_id

    def __getitem__(self, index):
        if self.return_id:
            return self.datas[index], self.labels[index], self.x[index], self.y[index], index
        else:
            return self.datas[index], self.labels[index], self.x[index], self.y[index]
    def __len__(self):
        return self.datas.shape[0]
def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    # adj = sp.coo_matrix(adj)
    # adj_ = adj + sp.eye(adj.shape[0])
    # rowsum = np.array(adj_.sum(1))
    # degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_mx_to_torch_sparse_tensor(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj)

def construct_graph_by_feature(x, k=25, mode= "connectivity", metric="correlation", include_self=False, n_components=None):
    if n_components is not None:
        pca = PCA(n_components=n_components)
        x = pca.fit_transform(x)
    feature_graph = kneighbors_graph(x, k, mode=mode, metric=metric, include_self=include_self)
    return feature_graph

def construct_graph_by_coordinate(cell_position, n_neighbors=25):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, class_balance, batch_size, num_workers, type):
    
    # split common | train_private | test_private
    common_y = np.intersect1d(train_y, test_y)
    train_private_y = np.setdiff1d(train_y, common_y)
    test_private_y = np.setdiff1d(test_y, common_y)

    print(f"common: {common_y}")
    print(f"train private: {train_private_y}")
    print(f"test_private: {test_private_y}")

    # to digit
    cell_type_dict = {}
    inverse_dict = {}
    cnt = 0

    for y in common_y:
        cell_type_dict[y] = cnt
        inverse_dict[cnt] = y
        cnt += 1

    for y in train_private_y:
        cell_type_dict[y] = cnt
        inverse_dict[cnt] = y
        cnt += 1

    for y in test_private_y:
        cell_type_dict[y] = cnt
        inverse_dict[cnt] = y
        cnt += 1

    train_y = np.array([cell_type_dict[x] for x in train_y])
    test_y = np.array([cell_type_dict[x] for x in test_y])

    # make classes set
    a, b, c = common_y.shape[0], train_private_y.shape[0], test_private_y.shape[0]
    common_classes = [i for i in range(a)]
    source_private_classes = [i + a for i in range(b)]
    target_private_classes = [i + a + b for i in range(c)]

    source_classes = common_classes + source_private_classes
    target_classes = common_classes + target_private_classes

    # target-private label
    tp_classes = sorted(set(target_classes) - set(source_classes))
    # source-private label
    sp_classes = sorted(set(source_classes) - set(target_classes))
    # common label
    common_classes = sorted(set(source_classes) - set(sp_classes))

    classes_set = {
    'source_classes': source_classes,
    'target_classes': target_classes,
    'tp_classes': tp_classes,
    'sp_classes': sp_classes,
    'common_classes': common_classes
    }
    
    if type == 'st2st':
        cell_position_train = []
        for pos in range(len(labeled_pos[0])):
            pos_x = labeled_pos[0][pos]
            pos_y = labeled_pos[1][pos]
            cell_position_train.append([pos_x,pos_y])
        cell_position_test = []
        for pos in range(len(unlabeled_pos[0])):
            pos_x = unlabeled_pos[0][pos]
            pos_y = unlabeled_pos[1][pos]
            cell_position_test.append([pos_x,pos_y])
        adj_spatial_train = construct_graph_by_coordinate(cell_position_train)
        adj_spatial_train = transform_adjacent_matrix(adj_spatial_train)
        adj_spatial_test = construct_graph_by_coordinate(cell_position_test)
        adj_spatial_test = transform_adjacent_matrix(adj_spatial_test)
        adj_matrix_train = adj_spatial_train
        adj_matrix_test = adj_spatial_test
    elif type == 'sc2sc':
        feature_graph_train = construct_graph_by_feature(train_X)
        feature_graph_test = construct_graph_by_feature(test_X)
        # feature_graph_train = transform_adjacent_matrix(feature_graph_train)
        # feature_graph_test = transform_adjacent_matrix(feature_graph_test)
        adj_matrix_train = feature_graph_train
        adj_matrix_test = feature_graph_test
    elif type == 'sc2st':
        cell_position_test = []
        for pos in range(len(unlabeled_pos[0])):
            pos_x = unlabeled_pos[0][pos]
            pos_y = unlabeled_pos[1][pos]
            cell_position_test.append([pos_x,pos_y])
        
        adj_spatial_test = construct_graph_by_coordinate(cell_position_test, n_neighbors=5)
        adj_spatial_test = transform_adjacent_matrix(adj_spatial_test)
        adj_matrix_test = adj_spatial_test
        print('test done!')

        ###### ADD feature !!!!!!!! ###########
        # feature_graph_test = construct_graph_by_feature(test_X, n_components=50)
        # adj_matrix_test = adj_spatial_test.toarray() * feature_graph_test.toarray()
        # adj_matrix_test = coo_matrix(adj_matrix_test)

        feature_graph_train = construct_graph_by_feature(train_X, k=5, n_components=50)
        # feature_graph_train = transform_adjacent_matrix(feature_graph_train)
        # feature_graph_test = transform_adjacent_matrix(feature_graph_test)
        adj_matrix_train = feature_graph_train
        print('train done!')


    # make dataset and dataloader
    uniformed_index = len(classes_set['source_classes'])

    source_train_ds = ST_Dataset(train_X, train_y, adj_matrix_train,labeled_pos[0], labeled_pos[1])
    target_train_ds = ST_Dataset(test_X, test_y, adj_matrix_test,unlabeled_pos[0], unlabeled_pos[1])

    source_test_ds = ST_Dataset(train_X, train_y, adj_matrix_train,labeled_pos[0], labeled_pos[1])
    target_test_ds = ST_Dataset(test_X, test_y, adj_matrix_test,unlabeled_pos[0], unlabeled_pos[1])

    print('st_dataset done!')
    # balanced sampler for source train
    classes = source_train_ds.labels
    freq = Counter(classes)
    class_weight = {x : 1.0 / freq[x] if class_balance else 1.0 for x in freq}

    source_weights = [class_weight[x] for x in source_train_ds.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

    print('sampler done!')
    source_train_dl = DataLoader(dataset=source_train_ds, batch_size=batch_size,
                             sampler=sampler, num_workers=num_workers, drop_last=True)
    
    target_train_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)
    target_test_dl = DataLoader(dataset=target_test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)

    print('dataloader done!')
    # for memory queue init
    target_initMQ_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True)
    # for tsne feature visualization
    source_test_dl = DataLoader(dataset=source_test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    
    print('memory queue done!')
    # feature_graph_train = preprocess_graph(feature_graph_train)
    # feature_graph_test = preprocess_graph(feature_graph_test)
    # adj_spatial_train = preprocess_graph(adj_spatial_train)
    # adj_spatial_test = preprocess_graph(adj_spatial_test)
    adj_matrix_train = preprocess_graph(adj_matrix_train)
    adj_matrix_test = preprocess_graph(adj_matrix_test)
    
    print('preprocess graph done!')
    # feature_graph_train = torch.from_numpy(feature_graph_train.A)
    # feature_graph_test = torch.from_numpy(feature_graph_test.A)
    # adj_spatial_train = torch.from_numpy(adj_spatial_train.A)
    # adj_spatial_test = torch.from_numpy(adj_spatial_test.A)
    
    graph = {
        'source': adj_matrix_train,
        'target': adj_matrix_test,
    }
    
    return classes_set, train_X.shape[1], source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl, graph ,inverse_dict

def load_data(args):
    adata_train = sc.read(args.train_dataset)
    adata_test = sc.read(args.test_dataset)
    if sp.issparse(adata_train.X):
        train_X = adata_train.X.toarray()
    else:
        train_X = adata_train.X

    if sp.issparse(adata_test.X):
        test_X = adata_test.X.toarray()
    else:
        test_X = adata_test.X
    
    train_y = adata_train.obs['cell_type']
    test_y = adata_test.obs['cell_type']
    
    if args.type=="sc2sc":
        labeled_pos = []
        labeled_pos.append([0 for _ in range(adata_train.shape[0])])
        labeled_pos.append([0 for _ in range(adata_train.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(adata_test.shape[0])])
        unlabeled_pos.append([0 for _ in range(adata_test.shape[0])])
    elif args.type=="st2st":
        labeled_pos = []
        labeled_pos.append(adata_train.obsm['pos'][:, 0])
        labeled_pos.append(adata_train.obsm['pos'][:, 1])
        unlabeled_pos = []
        unlabeled_pos.append(adata_test.obsm['pos'][:, 0])
        unlabeled_pos.append(adata_test.obsm['pos'][:, 1])
    else:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(adata_train.shape[0])])
        labeled_pos.append([0 for _ in range(adata_train.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append(adata_test.obsm['pos'][:, 0])
        unlabeled_pos.append(adata_test.obsm['pos'][:, 1])
    
    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_Hubmap_CL_intra_data(args):

    train_df = pd.read_csv(data_path+"/Hubmap_CL_intra_0.5/train.csv")
    test_df = pd.read_csv(data_path+"/Hubmap_CL_intra_0.5/test.csv")

    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    train_y = train_df['cell_type_A']
    test_y = test_df['cell_type_A']

    labeled_pos = train_df.iloc[:, -6:-4].values.T # x,y coordinates, indexes depend on specific datasets
    unlabeled_pos = test_df.iloc[:, -6:-4].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Hubmap_SB_intra_data(args):

    train_df = pd.read_csv(data_path+"/Hubmap_SB_intra/train.csv")
    test_df = pd.read_csv(data_path+"/Hubmap_SB_intra/test.csv")

    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    train_y = train_df['cell_type_A']
    test_y = test_df['cell_type_A']

    labeled_pos = train_df.iloc[:, -6:-4].values.T # x,y coordinates, indexes depend on specific datasets
    unlabeled_pos = test_df.iloc[:, -6:-4].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Lung_intra_data(args):

    train_adata = sc.read_h5ad(data_path+"/Lung_intra_0.5/train.h5ad")
    test_adata = sc.read_h5ad(data_path+"/Lung_intra_0.5/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['cell type'].values
    test_y = test_adata.obs['cell type'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['X_centroid'].values)
    labeled_pos.append(train_adata.obs['Y_centroid'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['X_centroid'].values)
    unlabeled_pos.append(test_adata.obs['Y_centroid'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Hyp_intra_data(args):

    train_df = pd.read_csv(data_path+"/Hyp_intra_0.5/train.csv")
    test_df = pd.read_csv(data_path+"/Hyp_intra_0.5/test.csv")

    train_X = train_df.iloc[:, 10:].values
    test_X = test_df.iloc[:, 10:].values
    train_y = train_df["Cell_class"].values
    test_y = test_df["Cell_class"].values

    labeled_pos = train_df.iloc[:, 6:8].values.T
    unlabeled_pos = test_df.iloc[:, 6:8].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Spe_Diabetes_intra_data(args):

    train_adata = sc.read_h5ad(data_path+"/Spe_Diabetes_intra_0.5/train.h5ad")
    test_adata = sc.read_h5ad(data_path+"/Spe_Diabetes_intra_0.5/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['pos_x'].values)
    labeled_pos.append(train_adata.obs['pos_y'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['pos_x'].values)
    unlabeled_pos.append(test_adata.obs['pos_y'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Spe_WT_intra_data(args):

    train_adata = sc.read_h5ad(data_path+"/Spe_WT_intra_0.5/train.h5ad")
    test_adata = sc.read_h5ad(data_path+"/Spe_WT_intra_0.5/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['pos_x'].values)
    labeled_pos.append(train_adata.obs['pos_y'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['pos_x'].values)
    unlabeled_pos.append(test_adata.obs['pos_y'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Tonsil_BE_cross_data(args):
    # filename = "/data/user/luomai/UniOT-for-UniDA/data/CODEX/BE_Tonsil_l3_dryad.csv"
    
    # df = pd.read_csv(filename)
    # train_df = df.loc[df['sample_name'] == 'tonsil']
    # test_df = df.loc[df['sample_name'] == 'Barretts Esophagus']

    train_df = pd.read_csv("/data/user/luomai/UniOT-for-UniDA/STOT/data/Tonsil_BE_cross/train.csv")
    test_df = pd.read_csv("/data/user/luomai/UniOT-for-UniDA/STOT/data/Tonsil_BE_cross/test.csv")

    train_X = train_df.iloc[:, 1:-4].values
    test_X = test_df.iloc[:, 1:-4].values
    train_y = train_df['cell_type'].str.lower()
    test_y = test_df['cell_type'].str.lower()

    labeled_pos = train_df.iloc[:, -4:-2].values.T
    unlabeled_pos = test_df.iloc[:, -4:-2].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_BE_Tonsil_cross_data(args):
    # filename = "/data/user/luomai/UniOT-for-UniDA/data/CODEX/BE_Tonsil_l3_dryad.csv"
    
    # df = pd.read_csv(filename)
    # train_df = df.loc[df['sample_name'] == 'Barretts Esophagus']
    # test_df = df.loc[df['sample_name'] == 'tonsil']

    train_df = pd.read_csv("/data/user/luomai/UniOT-for-UniDA/STOT/data/BE_Tonsil_cross/train.csv")
    test_df = pd.read_csv("/data/user/luomai/UniOT-for-UniDA/STOT/data/BE_Tonsil_cross/test.csv")

    train_X = train_df.iloc[:, 1:-4].values
    test_X = test_df.iloc[:, 1:-4].values
    train_y = train_df['cell_type'].str.lower()
    test_y = test_df['cell_type'].str.lower()

    labeled_pos = train_df.iloc[:, -4:-2].values.T
    unlabeled_pos = test_df.iloc[:, -4:-2].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Hubmap_CL_cross_data(args):

    train_df = pd.read_csv(data_path+"/Hubmap_CL_cross/train.csv")
    test_df = pd.read_csv(data_path+"/Hubmap_CL_cross/test.csv")

    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    train_y = train_df['cell_type_A']
    test_y = test_df['cell_type_A']

    labeled_pos = train_df.iloc[:, -6:-4].values.T # x,y coordinates, indexes depend on specific datasets
    unlabeled_pos = test_df.iloc[:, -6:-4].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Hubmap_SB_cross_data(args):

    train_df = pd.read_csv(data_path+"/Hubmap_SB_cross/train.csv")
    test_df = pd.read_csv(data_path+"/Hubmap_SB_cross/test.csv")

    train_X = train_df.iloc[:, 1:49].values
    test_X = test_df.iloc[:, 1:49].values
    train_y = train_df['cell_type_A']
    test_y = test_df['cell_type_A']

    labeled_pos = train_df.iloc[:, -6:-4].values.T # x,y coordinates, indexes depend on specific datasets
    unlabeled_pos = test_df.iloc[:, -6:-4].values.T

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Spe_Diabetes_cross_data(args):

    train_adata = sc.read_h5ad(data_path+"/Spe_Diabetes_cross/train.h5ad")
    test_adata = sc.read_h5ad(data_path+"/Spe_Diabetes_cross/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['pos_x'].values)
    labeled_pos.append(train_adata.obs['pos_y'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['pos_x'].values)
    unlabeled_pos.append(test_adata.obs['pos_y'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def load_Spe_WT_cross_data(args):

    train_adata = sc.read_h5ad(data_path+"/Spe_WT_cross/train.h5ad")
    test_adata = sc.read_h5ad(data_path+"/Spe_WT_cross/test.h5ad")

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values

    labeled_pos = []
    labeled_pos.append(train_adata.obs['pos_x'].values)
    labeled_pos.append(train_adata.obs['pos_y'].values)
    unlabeled_pos = []
    unlabeled_pos.append(test_adata.obs['pos_x'].values)
    unlabeled_pos.append(test_adata.obs['pos_y'].values)

    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers)

def preprocess(adata):
    count = Counter(adata.obs['CellType'])
    for key in count.keys():
        if count[key] < 100:
            adata = adata[adata.obs['CellType'] != key]
    return adata

def load_Immune_ALL_human_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/Immune_ALL_human/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/Immune_ALL_human/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X.A
    test_X = test_adata.X.A
    train_y = train_adata.obs['final_annotation'].values
    test_y = test_adata.obs['final_annotation'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)


    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)


def load_HumanPBMC_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/HumanPBMC/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/HumanPBMC/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X.A
    test_X = test_adata.X.A
    train_y = train_adata.obs['cell_type'].values
    test_y = test_adata.obs['cell_type'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)


    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_MCA_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/MCA/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/MCA/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X.A
    test_X = test_adata.X.A
    train_y = train_adata.obs['cell_type'].values
    test_y = test_adata.obs['cell_type'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)


    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_Pancrm_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/Pancrm/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/Pancrm/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X.A
    test_X = test_adata.X.A
    train_y = train_adata.obs['cell_type'].values
    test_y = test_adata.obs['cell_type'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)


    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_dataset5_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/dataset5/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/dataset5/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['cell_type'].values
    test_y = test_adata.obs['cell_type'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)


    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_pancreas_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/pancreas/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/pancreas/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['celltype'].values
    test_y = test_adata.obs['celltype'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)


    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_Muscle_M10x_MCA_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/Muscle_M10x_MCA/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/Muscle_M10x_MCA/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X.A
    test_X = test_adata.X.A
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)
    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_Lung_M10x_MCA_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/Lung_M10x_MCA/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/Lung_M10x_MCA/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X.A
    test_X = test_adata.X.A
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)
    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_HCA_AdultBoneMarrow_MCA_AdultBoneMarrow_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/HCA_AdultBoneMarrow_MCA_AdultBoneMarrow/train.h5ad")
    # train_adata = preprocess(train_adata)
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_Graph_singlecell/data/HCA_AdultBoneMarrow_MCA_AdultBoneMarrow/test.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X.A
    test_X = test_adata.X.A
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values
    if args.type:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    else:
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)
    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_sc2st_SpeWT_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    # train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_sc2st/data/reference_rna_hvg1k_sample.h5ad")
    train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_sc2st/data/reference_rna.h5ad")
    # train_adata = preprocess(train_adata)
    # test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_sc2st/data/WT3_hvg1k.h5ad")
    test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_sc2st/data/WT3.h5ad")
    # test_adata = preprocess(test_adata)

    train_X = train_adata.X
    test_X = test_adata.X
    train_y = train_adata.obs['CellType'].values
    test_y = test_adata.obs['CellType'].values
    if args.type=="sc2sc":
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    elif args.type=="st2st":
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)
    else:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['x'].values)
        unlabeled_pos.append(test_adata.obs['y'].values)
    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_sc2st_sperma_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    # train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_sc2st/data/reference_rna_hvg1k_sample.h5ad")
    train_adata = sc.read_h5ad("/data/chenyz/project/STOT/dataset/sperma/scRNA/sperma_sc.h5ad")
    # train_adata = preprocess(train_adata)
    # test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_sc2st/data/WT3_hvg1k.h5ad")
    test_adata = sc.read_h5ad("/data/chenyz/project/STOT/dataset/sperma/ST/sperma_st.h5ad")
    # test_adata = preprocess(test_adata)

    sc.pp.normalize_total(train_adata)
    sc.pp.normalize_total(test_adata)
    sc.pp.log1p(train_adata)
    sc.pp.log1p(test_adata)
    sc.pp.highly_variable_genes(train_adata, n_top_genes=6000)
    sc.pp.highly_variable_genes(test_adata, n_top_genes=6000)

    train_hvgs = train_adata.var[train_adata.var['highly_variable']].index
    test_hvgs = test_adata.var[test_adata.var['highly_variable']].index

    common_hvgs = train_hvgs.intersection(test_hvgs)
    common_hvgs = common_hvgs[:3000]
    print(f'select {len(common_hvgs)} common_hvgs!')

    train_adata = train_adata[:, common_hvgs]
    test_adata = test_adata[:, common_hvgs]

    train_X = train_adata.X.toarray()
    test_X = test_adata.X.toarray()
    train_y = train_adata.obs['cell_type'].values
    test_y = test_adata.obs['cell_type'].values
    if args.type=="sc2sc":
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    elif args.type=="st2st":
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)
    else:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obsm['xy'][:, 0])
        unlabeled_pos.append(test_adata.obsm['xy'][:, 1])
    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)

def load_sc2st_MOP_data(args):
    # adata = sc.read_h5ad("/bigdat2/user/zengys/zengys/data/singlecell/batch_effects/hd5_data/HCA_AdultOmentum_MCA_AdultOmentum.h5ad")
    # train_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_sc2st/data/reference_rna_hvg1k_sample.h5ad")
    train_adata = sc.read_h5ad("/data/chenyz/project/STOT/dataset/MOP/scRNA/scMOP.h5ad")
    # train_adata = preprocess(train_adata)
    # test_adata = sc.read_h5ad("/bigdat2/user/shanggny/project/STOT_sc2st/data/WT3_hvg1k.h5ad")
    test_adata = sc.read_h5ad("/data/chenyz/project/STOT/dataset/MOP/ST/MOP_st.h5ad")
    #### slice153
    test_adata = test_adata[test_adata.obs['slice_id']=='mouse1_slice153']
    # test_adata = preprocess(test_adata)
    train_X = train_adata.X.toarray()
    test_X = test_adata.X
    train_y = train_adata.obs['cell_type'].values
    test_y = test_adata.obs['cell_type'].values
    if args.type=="sc2sc":
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
        unlabeled_pos.append([0 for _ in range(test_adata.shape[0])])
    elif args.type=="st2st":
        labeled_pos = []
        labeled_pos.append(train_adata.obs['pos_x'].values)
        labeled_pos.append(train_adata.obs['pos_y'].values)
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)
    else:
        labeled_pos = []
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        labeled_pos.append([0 for _ in range(train_adata.shape[0])])
        unlabeled_pos = []
        unlabeled_pos.append(test_adata.obs['pos_x'].values)
        unlabeled_pos.append(test_adata.obs['pos_y'].values)
    return make_dataloader(train_X, test_X, train_y, test_y, labeled_pos, unlabeled_pos, args.class_balance, args.batch_size, args.num_workers, args.type)