import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torchsummary import summary
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# MultiHeadAttention
class MultiHeadSelfAttention(nn.Module):
    dim_in:int
    dim_k:int
    dim_v:int
    num_heads:int
    def __init__(self,dim_in,dim_k,dim_v,num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k%num_heads==0 and dim_v%num_heads==0
        self.dim_in=dim_in
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.num_heads=num_heads
        self.linear_q=nn.Linear(dim_in,dim_k,bias=False)
        self.linear_k=nn.Linear(dim_in,dim_k,bias=False)
        self.linear_v=nn.Linear(dim_in,dim_v,bias=False)
        self._norm_fact=1/sqrt(dim_k//num_heads)

    def forward(self,x):
        # x: tensor of shape (batch, n, dim_in)
        batch,n,dim_in=x.shape
        #print("dim_in:",dim_in)

        assert dim_in==self.dim_in

        nh=self.num_heads
        dk=self.dim_k//nh
        dv=self.dim_v//nh

        q=self.linear_q(x).reshape(batch,n,nh,dk).transpose(1,2)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)

        dist=torch.matmul(q,k.transpose(2,3))*self._norm_fact
        dist=torch.softmax(dist,dim=-1)

        att=torch.matmul(dist,v)
        att=att.transpose(1,2).reshape(batch,n,self.dim_v)
        return att
