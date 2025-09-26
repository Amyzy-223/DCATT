import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv1dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):

    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data,
                                            p=2,
                                            dim=0,
                                            maxnorm=self.max_norm)
        return super(LinearWithConstraint, self).forward(x)

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
        self.mlp = Conv2dWithConstraint(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

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
        if isinstance(support,list):
            for a in support:  # support: list
                a = a[:, :, :, -x.size(3):]  # cut a to match the length of x
                x1 = self.nconv(x,a)
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = self.nconv(x1,a)
                    out.append(x2)
                    x1 = x2
        else:
            a = support
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DCATT(nn.Module):
    """
        input:  fnirsx, As
            fnirsx: tensor [batch_size, fnirs_dim, node_num, lenP]
            As: list [tensor,] * the number of functional connectivity matrices
                each tensor is [batch_size, node_num, node_num, lenP]
        output: tensor [batch_size, out_dim]
        """
    def __init__ (self, fusionFlag='concat', gcnFlag='gcn', tempFlag='att', timeProj='att', layer=2, lenP=4, lenQ=1,
                  fnirs_dim=5, node_num=12, ecg_dim=14, dropout=0.5, out_dim=3, hidden_dim=64, end_dim=1024, readout='mean'):
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
        self.BN = nn.BatchNorm2d(fnirs_dim, momentum=0.01, affine=True, eps=1e-3)
        self.start_fnirs = Conv2dWithConstraint(in_channels=fnirs_dim, out_channels=hidden_dim,kernel_size=(1,1))

        # fnirs ecg 空间建模
        self.gconv_fnirs = nn.ModuleList()
        self.dropout_fnirs_spat = nn.ModuleList()

        # 归一层
        self.norm_fnirs = nn.ModuleList()

        # position embedding
        self.pos_emb_fnirs = PositionalEncoding(hidden_dim=hidden_dim, mode='ecg', dropout=dropout, max_len=lenP+lenQ)

        # fnirs ecg 时间建模，自注意力，不同时间步间融合
        self.timeconv_fnirs = nn.ModuleList()

        for i in range(self.layer):
            self.gconv_fnirs.append(gcn(hidden_dim, hidden_dim, dropout, support_len=2))
            self.dropout_fnirs_spat.append(nn.Dropout(p=dropout))
            self.timeconv_fnirs.append(nn.MultiheadAttention(embed_dim=hidden_dim * node_num, num_heads=self.num_head, dropout=dropout, batch_first=True))
            self.norm_fnirs.append(nn.LayerNorm(hidden_dim * node_num))

        # 时序映射
        self.multiAtt_fnirs = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=self.num_head, dropout=dropout, batch_first=True)

        # decoder
        self.end_conv_1 = LinearWithConstraint(hidden_dim, end_dim, bias=True)
        self.last = LinearWithConstraint(end_dim, out_dim, bias=True)


    # '''
    def forward(self, fnirsx, adj):

        # start conv
        fnirsx = self.BN(fnirsx)
        fnirsx = self.start_fnirs(fnirsx)  # [batch_size, fnirs_dim, node_num, lenP] -> [batch_size, hidden_dim, node_num, lenP]

        ## transpose
        # [batch_size, hidden_dim, node_num, lenP] -> [batch_size, lenP, hidden_dim*node_num]
        fnirsx_0 = fnirsx.reshape(fnirsx.shape[0], -1, fnirsx.shape[-1]).transpose(1, 2)

        for i in range(self.layer):
            ## 时间建模
            fnirsx, _ = self.timeconv_fnirs[i](fnirsx_0, fnirsx_0, fnirsx_0)  # [batch_size, lenP, hidden_dim*node_num]

            ## transpose
            # [batch_size, lenP, hidden_dim*node_num] -> [batch_size, hidden_dim, node_num, lenP]
            fnirsx = fnirsx.transpose(1, 2).reshape(fnirsx.shape[0], -1, self.node_num, fnirsx.shape[1])

            ## 空间建模 # [batch_size, hidden_dim, node_num, lenP] -> [batch_size, hidden_dim, node_num, lenP]
            fnirsx = self.gconv_fnirs[i](fnirsx, adj)  # [batch_size, hidden_dim, node_num, lenP] -> [batch_size, hidden_dim, node_num, lenP]
            fnirsx = self.dropout_fnirs_spat[i](fnirsx)

            ## transpose
            # [batch_size, hidden_dim, node_num, lenP] -> [batch_size, lenP, hidden_dim*node_num]
            fnirsx = fnirsx.reshape(fnirsx.shape[0], -1, fnirsx.shape[-1]).transpose(1, 2)

            ## add & norm 在所有节点所有时间步上进行归一
            fnirsx_0 = self.norm_fnirs[i](fnirsx + fnirsx_0)  # [batch_size, lenP, hidden_dim*node_num]

        ## fnirs 全图读出
        # [batch_size, lenP, hidden_dim*node_num] -> [batch_size, hidden_dim, node_num, lenP] -> [batch_size, hidden_dim, lenP]
        fnirsx = fnirsx_0.transpose(1, 2).reshape(fnirsx.shape[0], -1, self.node_num, fnirsx.shape[1])
        fnirsx = torch.mean(fnirsx, dim=2, keepdim=True).squeeze(2)
        fnirsx = fnirsx.transpose(1, 2)  # [batch_size, hidden_dim, lenP] -> [batch_size, lenP, hidden_dim]

        ## 位置编码 并获取lenQ的Q
        fnirsx, pos_fnirs = self.pos_emb_fnirs(fnirsx)  # pos_fnirs: [1, lenP+lenQ, hidden_dim]

        # 时序预测  # ->[batch_size, lenQ, hidden_dim]
        fnirsx, _ = self.multiAtt_fnirs(query=pos_fnirs[:, -self.lenQ:, :].repeat(fnirsx.shape[0], 1, 1), key=fnirsx, value=fnirsx)

        out = fnirsx.squeeze()  # [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        out = self.end_conv_1(F.relu(out))  # [batch_size, hidden_dim] -> [batch_size, end_dim]
        out = self.last(F.relu(out))  # [batch_size, end_dim] -> [batch_size, out_dim]
        return out



