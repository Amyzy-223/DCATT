
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear


class PositionalEncoding(nn.Module):
    """
    位置编码
        in / out: [batch_size, node_num, lenP , num_hidden]
    """
    def __init__(self, hidden_dim, mode, dropout=0.3, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        pe = torch.zeros((max_len, hidden_dim))  # 生成位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, hidden_dim, 2, dtype=torch.float) * -(math.log(10000.0) / hidden_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        if mode == 'fnirs':  # 输入为4维张量，[batch_size, node_num, lenP, hidden_dim]，在前两维上操作相同，相加
            pe = pe.unsqueeze(0).unsqueeze(0)
        elif mode == 'ecg':  # 输入为3维张量，在batch上操作相同，相加
            pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        if self.mode == 'fnirs':
            x = x.transpose(1, 2)  # [batch_size, lenP, node_num, hidden_dim] -> [batch_size, node_num, lenP, hidden_dim]
            x = x + Variable(self.pe[:, :, :x.size(2), :],requires_grad=False).to(x.device)
            x = x.transpose(1, 2)
        elif self.mode == 'ecg':  # [batch_size, lenP, hidden_dim]
            x = x + Variable(self.pe[:, :x.size(1), :],requires_grad=False).to(x.device)
        self.pe = self.pe.to(x.device)
        return self.dropout(x), self.pe

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    """
    in / out: [batch_size, hidden_dim, node_nums, lenP]
    """
    def __init__(self,c_in,c_out,dropout,support_len=4,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:  # support: list
            a = a[:, :, :, -x.size(3):]  # cut a to match the length of x
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DCATT(nn.Module):
    """
    input:  fnirsx, As, ecg
        fnirsx: tensor [batch_size, fnirs_dim, node_num, lenP]
        As: list [tensor,] * the number of functional connectivity matrices
            each tensor is [batch_size, node_num, node_num, lenP]
        ecg: tensor [batch_size, ecg_dim, lenP]
    output: tensor [batch_size, out_dim, lenQ]
    """
    def __init__ (self, fusionFlag, gcnFlag='gcn', tempFlag='att', timeProj='att', layer=2, lenP=6, lenQ=4,
                  fnirs_dim=6, node_num=8, ecg_dim=14, dropout=0.3, out_dim=3, hidden_dim=256, end_dim=512, readout='mean'):
        super(DCATT, self).__init__()
        self.fusionFlag = fusionFlag
        self.gcnFlag = gcnFlag
        self.tempFlag = tempFlag
        self.timeProj = timeProj
        self.lenQ = lenQ
        self.layer = layer
        self.node_num = node_num
        self.num_head = 8
        self.readout = readout

        # 起始映射
        self.start_fnirs = nn.Conv2d(in_channels=fnirs_dim, out_channels=hidden_dim,kernel_size=(1,1))
        self.start_ecg = nn.Conv1d(in_channels=ecg_dim, out_channels=hidden_dim, kernel_size=1, bias=True)

        # fnirs ecg 空间建模
        self.gconv_fnirs = nn.ModuleList()
        self.gconv_ecg = nn.ModuleList()

        # 归一层
        self.norm_fnirs = nn.ModuleList()
        self.norm_ecg = nn.ModuleList()

        # position embedding
        self.pos_emb_fnirs = PositionalEncoding(hidden_dim=hidden_dim, mode='ecg', dropout=dropout, max_len=lenP + lenQ)
        self.pos_emb_ecg = PositionalEncoding(hidden_dim=hidden_dim, mode='ecg', dropout=dropout, max_len=lenP + lenQ)

        # fnirs ecg 时间建模，自注意力，不同时间步间融合
        self.timeconv_fnirs = nn.ModuleList()
        self.timeconv_ecg = nn.ModuleList()

        for i in range(self.layer):
            self.gconv_fnirs.append(gcn(hidden_dim, hidden_dim, dropout))
            self.gconv_ecg.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, bias=True))
            self.timeconv_fnirs.append(nn.MultiheadAttention(embed_dim=hidden_dim * node_num, num_heads=8, dropout=dropout, batch_first=True))
            self.timeconv_ecg.append(nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True))
            self.norm_fnirs.append(nn.LayerNorm(hidden_dim * node_num))
            self.norm_ecg.append(nn.LayerNorm(hidden_dim))

        # 时序映射
        self.multiAtt_fnirs = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.multiAtt_ecg = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)

        # decoder
        hidden_dim *= 2
        self.end_conv_1 = nn.Conv1d(in_channels=hidden_dim, out_channels=end_dim, kernel_size=1, bias=True)
        self.end_conv_2 = nn.Conv1d(in_channels=end_dim, out_channels=out_dim, kernel_size=1, bias=True)

    def forward(self, fnirsx, As, ecg):
        ## start conv
        fnirsx = self.start_fnirs(fnirsx)  # [batch_size, fnirs_dim, node_num, lenP] -> [batch_size, hidden_dim, node_num, lenP]
        ecg = self.start_ecg(ecg)  # [batch_size, ecg_dim, node_num, lenP] -> [batch_size, hidden_dim, node_num, lenP]

        ## transpose
        # [batch_size, hidden_dim, node_num, lenP] -> [batch_size, lenP, hidden_dim*node_num]
        fnirsx_0 = fnirsx.reshape(fnirsx.shape[0], -1, fnirsx.shape[-1]).transpose(1, 2)
        ecg_0 = ecg.transpose(1, 2)  # [batch_size, hidden_dim, lenP] -> [batch_size, lenP, hidden_dim]

        for i in range(self.layer):
            ## 时间建模
            fnirsx, _ = self.timeconv_fnirs[i](fnirsx_0, fnirsx_0, fnirsx_0)  # [batch_size, lenP, hidden_dim*node_num]
            ecg, _ = self.timeconv_ecg[i](ecg_0, ecg_0, ecg_0)  # [batch_size, lenP, hidden_dim]

            ## transpose
            # [batch_size, lenP, hidden_dim*node_num] -> [batch_size, hidden_dim, node_num, lenP]
            fnirsx = fnirsx.transpose(1, 2).reshape(fnirsx.shape[0], -1, self.node_num, fnirsx.shape[1])
            ecg = ecg.transpose(1, 2)  # [batch_size, hidden_dim, lenP]

            ## 空间建模 # [batch_size, hidden_dim, node_num, lenP] -> [batch_size, hidden_dim, node_num, lenP]
            fnirsx = self.gconv_fnirs[i](fnirsx, As)  # [batch_size, hidden_dim, node_num, lenP] -> [batch_size, hidden_dim, node_num, lenP]
            ecg = self.gconv_ecg[i](ecg)  # [batch_size, hidden_dim, lenP] -> [batch_size, hidden_dim, lenP]


            ## transpose
            # [batch_size, hidden_dim, node_num, lenP] -> [batch_size, lenP, hidden_dim*node_num]
            fnirsx = fnirsx.reshape(fnirsx.shape[0], -1, fnirsx.shape[-1]).transpose(1, 2)
            ecg = ecg.transpose(1, 2)  # [batch_size, hidden_dim, lenP] -> [batch_size, lenP, hidden_dim]

            ## add & norm 在所有节点所有时间步上进行归一
            fnirsx_0 = self.norm_fnirs[i](fnirsx + fnirsx_0)  # [batch_size, lenP, hidden_dim*node_num]
            ecg_0 = self.norm_ecg[i](ecg + ecg_0)  # [batch_size, lenP, hidden_dim]

        ## fnirs 全图读出
        # [batch_size, lenP, hidden_dim*node_num] -> [batch_size, hidden_dim, node_num, lenP] -> [batch_size, hidden_dim, lenP]
        fnirsx = fnirsx_0.transpose(1, 2).reshape(fnirsx.shape[0], -1, self.node_num, fnirsx.shape[1])
        fnirsx = torch.mean(fnirsx, dim=2, keepdim=True).squeeze(2)
        fnirsx = fnirsx.transpose(1, 2)  # [batch_size, hidden_dim, lenP] -> [batch_size, lenP, hidden_dim]

        ## 位置编码 并获取lenQ的Q
        fnirsx, pos_fnirs = self.pos_emb_fnirs(fnirsx)  # pos_fnirs: [1, lenP+lenQ, hidden_dim]
        ecg, pos_ecg = self.pos_emb_ecg(ecg_0)  # pos_fnirs: [1, lenP+lenQ, hidden_dim]

        fnirsx, _ = self.multiAtt_fnirs(query=pos_fnirs[:, -self.lenQ:, :].repeat(fnirsx.shape[0], 1, 1),
                                        key=fnirsx, value=fnirsx)
        ecg, _ = self.multiAtt_ecg(query=pos_ecg[:, -self.lenQ:, :].repeat(ecg.shape[0], 1, 1), key=ecg, value=ecg)

        # concat
        out = torch.cat((fnirsx, ecg), dim=-1)  # -> [batch_size, lenQ, 2*hidden_dim]

        out = F.relu(out.transpose(1, 2))  # [batch_size, lenQ, hidden_dim] -> [batch_size, hidden_dim, lenQ]
        out = F.relu(self.end_conv_1(out))  # [batch_size, hidden_dim, lenQ] -> [batch_size, end_dim, lenQ]
        out = self.end_conv_2(out)  # [batch_size, end_dim, lenQ] -> [batch_size, out_dim, lenQ]
        return out
