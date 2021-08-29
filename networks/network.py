

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
        #print('x',x.shape)
        x = x.squeeze(-1)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        #print('attention', attention) #attention torch.Size([128, 25, 25])
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

class SPE_Module(Module):
    def __init__(self, in_dim):
        super(SPE_Module, self).__init__()
        self.chanel_in = in_dim
        #self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        
    def forward(self, x): #x:  torch.Size([128, 24, 11, 11, 2])
        m_batchsize, C, height, width, channle = x.size()
        #print('channle', channle)
        proj_query = x.view(m_batchsize, -1, channle)
        # print('proj_query', proj_query.shape) #proj_query torch.Size([128, 24, 50])
        proj_key = x.view(m_batchsize, -1, channle).permute(0, 2, 1) #proj_key torch.Size([128, 50, 24])
        # print('proj_key', proj_key.shape)
        energy = torch.bmm(proj_key, proj_query)
        # print('energy', energy.shape)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # print('energy_new', energy_new.shape)

        # attention = self.softmax(energy_new)
        attention = self.softmax(energy)

        #print('attention',attention.shape)
        proj_value = proj_query #x.view(m_batchsize, C, -1)
        out = torch.bmm(proj_value,attention) # out torch.Size([128, 24, 50])
        # print('out', out.shape)
        out = out.view(m_batchsize, C, height, width, channle)
        # out = self.gamma*out + x 
        # out = self.gamma*out
        # out = out.unsqueeze(-1)
        # print('out', out.shape)
        return out
#s
#class Attention(nn.Module):
    #"""
    #Attention network for calculate attention value
    #"""
    #def __init__(self, encoder_dim, decoder_dim, attention_dim):
        #"""
        #:param encoder_dim: input size of encoder network
        #:param decoder_dim: input size of decoder network
        #:param attention_dim: input size of attention network
        #"""
        #super(Attention, self).__init__()
        #self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        #self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        #self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        #self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    #def forward(self, encoder_out, decoder_hidden):
        #m_batchsize, C, height, width, channle = x.size() #128 24 5 5 20
        #proj_query = x.view(m_batchsize, C,height, width, -1) #128 600 20
        
        #att1 = self.encoder_att(proj_query)  # (batch_size, num_pixels, attention_dim)
        #att2 = self.decoder_att(proj_query)  # (batch_size, attention_dim)
        #att = self.full_att(self.relu(att1 + att2)).squeeze(-1)
        #alpha = self.softmax(att)  # (batch_size, num_pixels)
        #attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        #return attention_weighted_encoding, alpha
class spectral_attention(nn.Module):
    """
    Learn cross band spectral features with a set of convolutions and spectral pooling attention layers
    The feature maps should be pooled to remove spatial dimensions before reading in the module
    Args:
        in_channels: number of feature maps of the current image
    """
    def __init__(self):
        super(spectral_attention, self).__init__()        
        # Weak Attention with adaptive kernel size based on size of incoming feature map
        #if filters == 22:
            #kernel_size = 3
        #elif filters == 64:
            #kernel_size = 5
        #elif filters == 128:
            #kernel_size = 7
        #else:
            #raise ValueError(
                #"Unknown incoming kernel size {} for attention layers".format(kernel_size))
        
        #self.attention_conv1 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding="same")
        #self.attention_conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding="same")
        
        #TODO Does this pool size change base on in_features?
        #self.fc1 = nn.Linear(in_features=filters, out_features=classes)
        
    def forward(self, x):

        m_batchsize, C, height, width, channle = x.size() #128 24 5 5 20
        filters, kernel_size = C, 3
        self.attention_conv1 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding="same").cuda()
        self.attention_conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding="same").cuda()
        permuted_input = x.permute(0, 2, 3,1,4)
        #print('permuted_input', permuted_input.shape)
        m_batchsize,height, width,C, channle = permuted_input.size()
        pooled_features = permuted_input.reshape(-1,C,channle)
        #proj_query = x.view(m_batchsize, C,height, width, -1) #128 600 20
        """Calculate attention and class scores for batch"""
        #Global pooling and add dimensions to keep the same shape
        #pooled_features = global_spectral_pool(x)
        
        #Attention layers
        attention = self.attention_conv1(pooled_features)
        attention = F.relu(attention)
        attention = self.attention_conv2(attention)
        attention = F.sigmoid(attention)
        #print('attention before', attention.shape)
        #Add dummy dimension to make the shapes the same
        #attention = attention.unsqueeze(-1)
        attention = pooled_features * attention
        #print('attention', attention.shape)
        attention = attention.reshape(m_batchsize,height, width,C, channle).permute(0,3,1,2,4)
        #print('attention', attention.shape)
        #print('attention after that', attention.shape)
        # Classification Head
        #pooled_attention_features = global_spectral_pool(attention)
        #pooled_attention_features = torch.flatten(pooled_attention_features, start_dim=1)
        #class_features = self.fc1(pooled_attention_features)
        #class_scores = F.softmax(class_features)
        
        return attention
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
        #kernel_3d = math.floor((band - 4) / 2)
        kernel_3d = 5
        kernel_stride = 2
        self.conv13 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(1, 1, kernel_3d), stride=(1, 1, kernel_stride))

        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(1, 1, band), stride=(1, 1, 1))
        self.batch_norm21 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.conv222 = nn.Conv3d(in_channels=6, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm222 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        
        self.conv223 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm223 = nn.Sequential(nn.BatchNorm3d(6, eps=0.001, momentum=0.1, affine=True), mish())
        
        
        
        
        
        self.conv224 = nn.Conv3d(in_channels=12, out_channels=6, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.conv225 = nn.Conv3d(in_channels=24, out_channels=24, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True), mish())
        
        #self.attention_spectral = CAM_Module(24)
        self.attention_spectral = spectral_attention()
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
        #X1 = torch.mul(X1, X13) #X1: torch.Size([8, 24, 11, 11, 2])
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
        
        #X2 = self.attention_spatial(X22)
        #X2 = torch.mul(X2, X22)
        X2= X22
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














