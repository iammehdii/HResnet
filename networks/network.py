
"""
Created on Wed Oct 21 21:08:41 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Sequential, Conv2d, Conv3d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding,ReLU


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    def forward(self, x):
        #a = torch.tanh(F.softplus(x))
        aa = F.softplus(x)
        #print('aaaaaaaaa mish',a.shape)
        #y = x * a
        m = ReLU()
        #print('yyyyyyyyy mish',y.shape)
        return m(x)


class PAM_Module(Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        
    def forward(self, x):
        print('x',x.shape)
        x = x.squeeze(-1)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        print('attention', attention) #attention torch.Size([128, 25, 25])
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        # out = (self.gamma*out + x).unsqueeze(-1)
        # out = (self.gamma*out ).unsqueeze(-1)
        out = out.unsqueeze(-1)
        #print('attention spatial',attention.shape)
        return out


class CAM_Module(Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        
    def forward(self, x): #x:  torch.Size([128, 24, 11, 11, 2])
        m_batchsize, C, height, width, channle = x.size()
        # print('channle', channle)
        proj_query = x.view(m_batchsize, C, -1)
        # print('proj_query', proj_query.shape) #proj_query torch.Size([128, 24, 50])
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) #proj_key torch.Size([128, 50, 24])
        # print('proj_key', proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)
        # print('energy', energy.shape)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # print('energy_new', energy_new.shape)

        # attention = self.softmax(energy_new)
        attention = self.softmax(energy)

        #print('attention',attention.shape)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value) # out torch.Size([128, 24, 50])
        # print('out', out.shape)
        out = out.view(m_batchsize, C, height, width, channle)
        # out = self.gamma*out + x 
        # out = self.gamma*out
        # out = out.unsqueeze(-1)
        # print('out', out.shape)
        return out


class HResNetAM(nn.Module):
    def __init__(self, band, classes):
        super(HResNetAM, self).__init__()
        self.name = 'HResNetAM'
        

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, 5), stride=(1, 1, 2))
        self.batch_norm11 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())

        self.conv122 = nn.Conv3d(in_channels=6, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm122 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        self.conv123 = nn.Conv3d(in_channels=12, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm123 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        self.conv124 = nn.Conv3d(in_channels=12, out_channels=6, padding=(0, 0, 2), kernel_size=(1, 1, 5), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        kernel_3d = math.floor((band - 4) / 2)
        #kernel_3d = 50
        self.conv13 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, band), stride=(1, 1, 1))
        self.batch_norm21 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.conv222 = nn.Conv3d(in_channels=6, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm222 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.conv223 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm223 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        
        
        
        
        
        self.conv224 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.conv225 = nn.Conv3d(in_channels=24, out_channels=24, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.attention_spectral = CAM_Module(24)
        self.attention_spatial = PAM_Module(24)
        
        self.batch_norm_spectral = nn.Sequential(nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), mish(), nn.Dropout(p=0.5))
        self.batch_norm_spatial = nn.Sequential(nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), mish(), nn.Dropout(p=0.5))
        
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(nn.Linear(48, classes))
        #self.dropout = Dropout3d(p=0.2)
        
    def forward(self, X):
        #print('X',X.shape)  #X torch.Size([128, 1, 11, 11, 103])

        X11 = self.conv11(X) # X11: torch.Size([128, 24, 11, 11, 50])
        #print('X11:',X11.shape)
        X11 = self.batch_norm11(X11)
        
        XS1 = torch.chunk(X11,4,dim=1)
        #print('XS1:',XS1.shape)
        
        
        ########### Hierarcical 
        X121 = XS1[0]  #0     ----------------------------------------------------------------------------->>>>>>>>>>>>.OUT
        X122 = self.conv122(XS1[1]) #X122: torch.Size([128, 6, 11, 11, 50]) 
        #print('X122:',X122.shape) 
        X122 = self.batch_norm122(X122) # ----------------------------------------------------------------->>>>>>>>>>>>OUT
        
        X123 = torch.cat((X122, XS1[2]), dim=1) #X123: torch.Size([128, 12, 11, 11, 50])
        #print('X123:',X123.shape)
        X123 = self.conv123(X123) #X123: torch.Size([128, 6, 11, 11, 50]) jjjjjjjjjjjj
        #print('X123:',X123.shape,'jjjjjjjjjjjj')
        X123 = self.batch_norm123(X123) #-------------------------------------------------------------------->>> out
        
        X124 = torch.cat((X123, XS1[3]), dim=1)
        
        X124 = self.conv124(X124)    #---------------------------------------------------------------------->>>>
        X124 = self.batch_norm123(X124)
        
        X12 = torch.cat((X121, X122, X123, X124), dim=1)
        #print('X12',X12.shape,'hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')

        X13 = self.conv13(X12)
        X13 = self.batch_norm12(X13)
        # print('X13',X13.shape) # torch.Size([8, 24, 11,11, 2])

        X1 = self.attention_spectral(X13) #X1: torch.Size([8, 24, 11, 11, 2])
        # print('X1:',X1.shape)
        # X1 = torch.mul(X1, X13) #X1: torch.Size([8, 24, 11, 11, 2])
        #X1 = X13
        # print('torch.mul:',X1.shape,'FFFFFFFFFFFFFFFFFFFFFFFF')
        X21 = self.conv21(X)
        X21 = self.batch_norm21(X21)

        XS2 = torch.chunk(X21,4,dim=1)
        X221 = XS2[0]
        X222 = self.conv222(XS2[1])
        X222 = self.batch_norm222(X222)
        
        X223 = torch.cat((X222, XS2[2]), dim=1)
        X223 = self.conv223(X223)
        X223 = self.batch_norm223(X223)
        
        X224 = torch.cat((X223, XS2[3]), dim=1)
        X224 = self.conv224(X224)
        X224 = self.batch_norm222(X224)
        #print('X224',X224.shape)
        X22 = torch.cat((X221, X222, X223, X224), dim=1)
        #print('X22',X22.shape)
        X22 = self.conv225(X22)
        X22 = self.batch_norm22(X22)
        
        
        
        
        
        
        #print('X22',X22.shape)
        
        X2 = self.attention_spatial(X22)
        # X2 = torch.mul(X2, X22)
        #X2= X22
        #print('X2:',X2.shape)
        X1 = self.batch_norm_spectral(X1)
        X1 = self.global_pooling(X1)
        # drop out
        X1 = X1.squeeze(-1).squeeze(-1).squeeze(-1)
        X2 = self.batch_norm_spatial(X2)
        #print('X2',X2.shape)
        # drop out
        
        X2= self.global_pooling(X2)
        X2 = X2.squeeze(-1).squeeze(-1).squeeze(-1)

        X_pre = torch.cat((X1, X2), dim=1)
        
        output = self.full_connection(X_pre)
        return output














