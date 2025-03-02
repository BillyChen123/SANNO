import torch
import torch.nn.functional as F
import torch.nn as nn
from transformer import MHAEncoder, MHAEncoderXY
from graphTransformer import GraphTransformer
class BaseFeatureExtractor(nn.Module):
    '''
    From https://github.com/thuml/Universal-Domain-Adaptation
    a base class for feature extractor
    '''
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)

class MLP(BaseFeatureExtractor):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_dim = hidden_dim
    # def forward(self, data):
    def forward(self, x):
        # x, edge_index = data.x, data.edge_index
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLPTrans(BaseFeatureExtractor):
    def __init__(self, input_dim, hidden_dim):
        super(MLPTrans, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.MHA = MHAEncoder(in_dim=hidden_dim, gap=4, n_heads=4, emb_dim=hidden_dim)
        self.output_dim = self.MHA.output_dim
        # self.MHA = TransformerEncoder(in_dim=hidden_dim, gap=8, n_heads=4, emb_dim=hidden_dim, ffn_dim=4*hidden_dim)
    # def forward(self, data):
    def forward(self, x):
        # x, edge_index = data.x, data.edge_index
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        residual = x
        x = self.MHA(x)
        x += residual
        return x
    
class MLPTransXY(BaseFeatureExtractor):
    def __init__(self, input_dim, hidden_dim):
        super(MLPTransXY, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.MHA = MHAEncoderXY(in_dim=hidden_dim, gap=4, n_heads=4, emb_dim=hidden_dim)
        self.output_dim = self.MHA.output_dim
        # self.MHA = TransformerEncoder(in_dim=hidden_dim, gap=8, n_heads=4, emb_dim=hidden_dim, ffn_dim=4*hidden_dim)
    # def forward(self, data):
    def forward(self, x, pos_x, pos_y):
        # x, edge_index = data.x, data.edge_index
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        residual = x
        x = self.MHA(x, pos_x, pos_y)
        x += residual
        return x

class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp 
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """
    def __init__(self, in_dim, out_dim, hidden_mlp=2048, feat_dim=256, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
                            nn.Linear(in_dim, hidden_mlp),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_mlp, feat_dim))
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls

    
class GTransXY(nn.Module):
    def __init__(self, in_channels, hidden_channels,num_layers = 3,output_dim=32, out_channels=32):
        super(GTransXY, self).__init__()
        self.GTF = GraphTransformer(in_channels, hidden_channels, out_channels, num_layers=1)
        self.GTF2 = GraphTransformer(in_channels, hidden_channels, out_channels, num_layers=num_layers)
        # self.fc = nn.Linear(out_channels, output_dim)
        # self.relu = nn.ReLU()
        self.output_dim = hidden_channels

    def forward(self, x, adj_sp):
        # res1,loss1 = self.GTF(x,adj_feat)
        res2,loss2 = self.GTF2(x,adj_sp)
        
        # x = torch.mul(res1,res2)
        # x = self.fc(res2)
        return res2,loss2
