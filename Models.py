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

'''
Application to cross-fertilization of two modalities
'''
class MultiHeadSelfAttention_cross(nn.Module):

    num_heads:int
    def  __init__(self,dim_in,dim_k,dim_v,num_heads=8):
        '''

        :param dim_in: Dimension of Query
        :param dim_k: Dimension of Key
        :param dim_v: Dimension of Value
        :param num_heads: int
        '''
        super(MultiHeadSelfAttention_cross, self).__init__()
        assert dim_k%num_heads==0 and dim_v%num_heads==0
        self.dim_in=dim_in
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.num_heads=num_heads
        self.linear_q=nn.Linear(dim_in,dim_k,bias=False)
        self.linear_k=nn.Linear(dim_in,dim_k,bias=False)
        self.linear_v=nn.Linear(dim_in,dim_v,bias=False)
        self._norm_fact=1/sqrt(dim_k//num_heads)


    def forward(self,x1,x2):
        '''

        :param x: Modal1
        :param y: Modal2
        :return:
        '''
        batch,n,dim_in=x1.shape
        #self.dim_in=dim_in
        #self.dim_k=dim_in
        #self.dim_v=dim_in
        assert dim_in==self.dim_in

        nh = self.num_heads
        dk = self.dim_k//nh
        dv = self.dim_v//nh
        # ----------------------------------------------------------------- #
        # The first type of modal extraction
        # ----------------------------------------------------------------- #
        # ① modality of q combined with another modality of (k, v)
        # x1 as query
        # x2 as key and value
        q_x1 = self.linear_q(x1).reshape(batch, n, nh, dk).transpose(1, 2)
        k_x2 = self.linear_k(x2).reshape(batch, n, nh, dk).transpose(1, 2)
        v_x2 = self.linear_v(x2).reshape(batch, n, nh, dv).transpose(1, 2)
        dist_x1 = torch.matmul(q_x1, k_x2.transpose(2, 3)) * self._norm_fact
        dist_x1 = torch.softmax(dist_x1, dim=-1)
        att_x1 = torch.matmul(dist_x1, v_x2)
        att_x1 = att_x1.transpose(1, 2).reshape(batch, n, self.dim_v)

        # ----------------------------------------------------------------- #
        # The second type of modal extraction
        # ----------------------------------------------------------------- #
        # ② modality of q combined with another modality of (k, v)
        # x2 as query
        # x1 as key and value
        q_x2 = self.linear_q(x2).reshape(batch, n, nh, dk).transpose(1, 2)
        k_x1 = self.linear_k(x1).reshape(batch, n, nh, dk).transpose(1, 2)
        v_x1 = self.linear_v(x1).reshape(batch, n, nh, dv).transpose(1, 2)
        dist_x2 = torch.matmul(q_x2, k_x1.transpose(2,3))*self._norm_fact
        dist_x2 = torch.softmax(dist_x2, dim=-1)
        att_x2 = torch.matmul(dist_x2, v_x1)
        att_x2 = att_x2.transpose(1, 2).reshape(batch, n, self.dim_v)

        return att_x1, att_x2
    
    
'''
VGG13 bimodal crossover network
'''
class VGG_13_2(nn.Module):
    '''
    VGG_13-1D
    Date：2023/3/22
    Author：Tianqi Fan
    '''
    def __init__(self,num_class):
        super(VGG_13_2, self).__init__()

        self.layer1=nn.Sequential(
            nn.Conv1d(1,64,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer2=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer3=nn.Sequential(
            nn.Conv1d(128,256,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,256,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer4=nn.Sequential(
            nn.Conv1d(256,512,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512,512,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer5=nn.Sequential(
            nn.Conv1d(512,512,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512,512,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )


        self.layer1_ = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer2_ = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer3_ = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer4_ = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer5_ = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.MaxPool1d(2, 2)
        )


        self.fc1=nn.Sequential(
            #nn.Flatten(),
            nn.Linear(8192,512),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc3=nn.Linear(256,num_class)

        self.multi_head1 = MultiHeadSelfAttention(64, 64, 64)
        self.multi_head2 = MultiHeadSelfAttention(128, 128, 128)
        self.multi_head3 = MultiHeadSelfAttention(256, 256, 256)
        self.multi_head4 = MultiHeadSelfAttention(512, 512, 512)
        self.multi_head5 = MultiHeadSelfAttention(512*2, 512*2, 512*2)


        self.multi_head_cross1=MultiHeadSelfAttention_cross(64,64,64)
        self.multi_head_cross2 = MultiHeadSelfAttention_cross(128, 128, 128)
        self.multi_head_cross3 = MultiHeadSelfAttention_cross(256, 256, 256)
        self.multi_head_cross4 = MultiHeadSelfAttention_cross(512, 512, 512)
        self.multi_head_cross5 = MultiHeadSelfAttention_cross(512, 512, 512)

        # 位置编码
        self.pos_embed1 = nn.Parameter(torch.zeros(128, 128, 64))
        self.pos_embed1_ = nn.Parameter(torch.zeros(128, 128, 64))
        self.pos_embed2 = nn.Parameter(torch.zeros(128, 64, 128))
        self.pos_embed2_ = nn.Parameter(torch.zeros(128, 64, 128))
        self.pos_embed3 = nn.Parameter(torch.zeros(128, 32, 256))
        self.pos_embed3_ = nn.Parameter(torch.zeros(128, 32, 256))
        self.pos_embed4 = nn.Parameter(torch.zeros(128, 16, 512))
        self.pos_embed4_ = nn.Parameter(torch.zeros(128, 16, 512))
        self.pos_embed5 = nn.Parameter(torch.zeros(128, 8, 512))
        self.pos_embed5_ = nn.Parameter(torch.zeros(128, 8, 512))




    def forward(self,x):
        x1=x[:,:256]
        x2=x[:,256:]
        x1=x1.reshape(-1,1,256)
        x2=x2.reshape(-1,1,256)
        x1_1 = self.layer1(x1)
        x2_1 = self.layer1_(x2)


        x1 = x1_1.transpose(1, 2)
        x2 = x2_1.transpose(1, 2)
        x1=x1+self.pos_embed1
        x2=x2+self.pos_embed1_
        attx1_1,attx2_1=self.multi_head_cross1(x1,x2)
        x1=x1+attx1_1
        x2=x2+attx2_1
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)


        x1_2 = self.layer2(x1)
        x2_2 = self.layer2_(x2)
        x1 = x1_2.transpose(1, 2)
        x2 = x2_2.transpose(1, 2)
        x1 = x1 + self.pos_embed2
        x2 = x2 + self.pos_embed2_
        attx1_2,attx2_2 = self.multi_head_cross2(x1, x2)
        x1 = x1 + attx1_2
        x2 = x2 + attx2_2
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)


        x1_3 = self.layer3(x1)
        x2_3 = self.layer3_(x2)
        x1 = x1_3.transpose(1, 2)
        x2 = x2_3.transpose(1, 2)
        x1 = x1 + self.pos_embed3
        x2 = x2 + self.pos_embed3_
        attx1_3,attx2_3 = self.multi_head_cross3(x1, x2)
        x1 = x1 + attx1_3
        x2 = x2 + attx2_3
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)


        x1_4 = self.layer4(x1)
        x2_4 = self.layer4_(x2)
        x1 = x1_4.transpose(1, 2)
        x2 = x2_4.transpose(1, 2)
        x1 = x1 + self.pos_embed4
        x2 = x2 + self.pos_embed4_
        attx1_4,attx2_4 = self.multi_head_cross4(x1, x2)
        x1 = x1 + attx1_4
        x2 = x2 + attx2_4
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)


        # git的第一次尝试111
        x1_5 = self.layer5(x1)
        x2_5 = self.layer5_(x2)
        x1 = x1_5.transpose(1, 2)
        x2 = x2_5.transpose(1, 2)
        x1 = x1 + self.pos_embed5
        x2 = x2 + self.pos_embed5_
        attx1_5,attx2_5 = self.multi_head_cross5(x1, x2)
        x1 = x1 + attx1_5
        x2 = x2 + attx2_5


        com1=x1_1.reshape(128,-1)+x1_2.reshape(128,-1)+x1_3.reshape(128,-1)+x1_4.reshape(128,-1)
        com2=x2_1.reshape(128,-1)+x2_2.reshape(128,-1)+x2_3.reshape(128,-1)+x2_4.reshape(128,-1)
        out = torch.cat([x1,x2],dim=-1)
        out = out.transpose(1, 2)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return com1,com2,out
