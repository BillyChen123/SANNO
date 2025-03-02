import torch 
from torch import nn
from torch import optim
from torch.utils import data as Data
import numpy as np
import math

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

##Positional Encoder Layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        ##the sine function is used to represent the odd-numbered sub-vectors
        pe[:, 0::2] = torch.sin(position * div_term)
        ##the cosine function is used to represent the even-numbered sub-vectors
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
  def __init__(self, in_dim, gap, n_heads=4, emb_dim=32, ffn_dim=256, dropout=0.1):
    super().__init__()
    self.in_dim = in_dim
    self.gap = gap
    self.num_patch = in_dim // gap
    self.patch_dim = self.num_patch * gap
    self.output_dim = emb_dim
    
    self.to_patch_embedding = nn.Sequential(
      nn.Linear(gap, emb_dim),
      nn.LayerNorm(emb_dim),
    )
    self.pe = PositionalEncoding(d_model=emb_dim, dropout=dropout)
    self.encoder = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            dim_feedforward=ffn_dim, 
            nhead=n_heads, 
            dropout=dropout, 
            batch_first=True,
        )

  def split_patch(self, x):
    patchs = torch.split(x[:, :self.patch_dim], self.gap, dim=1)
    patchs = torch.stack(patchs, dim=0)
    if self.patch_dim < self.in_dim:
      last_patch = x[:, -self.gap:]
      patchs = torch.cat((patchs, last_patch.unsqueeze(0)), dim=0)
    patchs = patchs.transpose(1,0)
    return patchs
  
  def forward(self, x):
    x = self.split_patch(x)
    x = self.to_patch_embedding(x)
    x = self.pe(x)
    x = self.encoder(x)
    x = x.mean(dim=1)
    return x


class MHAEncoder(nn.Module):
  def __init__(self, in_dim, gap, n_heads=4, emb_dim=32, ffn_dim=256, dropout=0.1):
    super().__init__()
    self.in_dim = in_dim
    self.gap = gap
    self.num_patch = in_dim // gap
    self.patch_dim = self.num_patch * gap

    # if self.patch_dim < self.in_dim:
    #   self.output_dim = (self.num_patch + 1) * emb_dim
    # else:
    #   self.output_dim = self.num_patch* emb_dim
    
    self.output_dim = emb_dim
    
    self.to_patch_embedding = nn.Sequential(
      nn.Linear(gap, emb_dim),
      nn.LayerNorm(emb_dim),
    )
    self.pe = PositionalEncoding(d_model=emb_dim, dropout=dropout)
    self.encoder = nn.MultiheadAttention(
            embed_dim=emb_dim, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True,
        )

  def split_patch(self, x):
    patchs = torch.split(x[:, :self.patch_dim], self.gap, dim=1)
    patchs = torch.stack(patchs, dim=0)
    if self.patch_dim < self.in_dim:
      last_patch = x[:, -self.gap:]
      patchs = torch.cat((patchs, last_patch.unsqueeze(0)), dim=0)
    patchs = patchs.transpose(1,0)
    return patchs
  
  def forward(self, x):
    x = self.split_patch(x)
    x = self.to_patch_embedding(x)
    x = self.pe(x)
    x, attn_weight = self.encoder(x, x, x)
    x = x.mean(dim=1)
    # x = torch.flatten(x, start_dim=1)
    return x
  

class MHAEncoderXY(nn.Module):
  def __init__(self, in_dim, gap, n_heads=4, emb_dim=32, ffn_dim=256, dropout=0.1):
    super().__init__()
    self.in_dim = in_dim
    self.gap = gap
    self.num_patch = in_dim // gap
    self.patch_dim = self.num_patch * gap
    
    self.output_dim = emb_dim
    
    self.to_patch_embedding = nn.Sequential(
      nn.Linear(gap, emb_dim),
      nn.LayerNorm(emb_dim),
    )

    # self.pe_enc = nn.Linear(2, emb_dim)
    self.pe_enc = nn.Embedding.from_pretrained(positionalencoding2d(emb_dim, 110, 100).flatten(1).T)

    # self.pe = PositionalEncoding(d_model=emb_dim, dropout=dropout)
    self.norm = nn.LayerNorm(emb_dim)
    self.dropout = nn.Dropout(dropout)

    self.encoder = nn.MultiheadAttention(
            embed_dim=emb_dim, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True,
        )

  def split_patch(self, x):
    patchs = torch.split(x[:, :self.patch_dim], self.gap, dim=1)
    patchs = torch.stack(patchs, dim=0)
    if self.patch_dim < self.in_dim:
      last_patch = x[:, -self.gap:]
      patchs = torch.cat((patchs, last_patch.unsqueeze(0)), dim=0)
    patchs = patchs.transpose(1,0)
    return patchs
  
  def forward(self, x, pos_x, pos_y):
    x = self.split_patch(x)
    x = self.to_patch_embedding(x)

    # pe_input = self.pe_enc(torch.stack([pos_x, pos_y], 1))

    pos_x = (pos_x * 100).long()
    pos_y = (pos_y * 100).long()
    pos_x[pos_x>=110] = 109
    pos_y[pos_y>=100] = 99
    pos_x[pos_x<0] = 0
    pos_y[pos_y<0] = 0
    pe_input = pos_x*100+pos_y
    pe_input = self.pe_enc(pe_input)

    x = torch.cat([pe_input.unsqueeze(1), x], 1)
    x = self.norm(self.dropout(x))
    # x = self.pe(x)
    x, attn_weight = self.encoder(x, x, x)
    x = x.mean(dim=1)
    return x

if __name__=="__main__":
  x = torch.Tensor(64, 9)   # (batch_size, seq_len, in_dim)
  # model = SelfAttention(1024, 64)
  model = TransformerEncoder(in_dim=9, gap=4)
  z = model(x)