import torch
import torch.nn as nn
import numpy as np
from icecream import ic

class Attention(nn.Module):
    def __init__(self,embed_size,dout_p,heads=8 , test = False):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.n_heads = embed_size // heads
        assert self.embed_size % self.heads == 0
        self.Q2d = nn.Linear(embed_size,embed_size)
        self.K2d = nn.Linear(embed_size,embed_size)
        self.V2d = nn.Linear(embed_size,embed_size)
        self.d2O = nn.Linear(embed_size,embed_size)
        if test:
            for name,parameter in self.Q2d.named_parameters():
                parameter.data.fill_(.1)
            for name,parameter in self.K2d.named_parameters():
                parameter.data.fill_(.1)
            for name,parameter in self.V2d.named_parameters():
                parameter.data.fill_(.1)
            for name,parameter in self.d2O.named_parameters():
                parameter.data.fill_(.1)
            dout_p = 0
        self.dropout = nn.Dropout(dout_p)
    def forward(self,Q,K,V,mask=None):
        B , T , embed_size = Q.shape
        Q = self.Q2d(Q)
        K = self.K2d(K)
        V = self.V2d(V)
        Q = Q.reshape(B , T , self.heads, self.n_heads)
        K = K.reshape(B , T , self.heads, self.n_heads)
        V = V.reshape(B , T , self.heads, self.n_heads)
        

        attention = torch.einsum('bqhd,bkhd->bhqk',[Q,K])
        attention = attention / np.sqrt(self.n_heads)
        if mask is not None:
            mask = mask.view(B,1,T)
            mask = mask.unsqueeze(1)#.unsqueeze(2)
            attention = attention.masked_fill(mask==0,-float('inf'))
        attention = torch.softmax(attention,dim=-1)

        out = torch.einsum('bhqk,bkhd->bqhd',[attention,V])
        out = self.dropout(out)
        out = out.contiguous().view(B,T,embed_size)
        out = self.d2O(out)
        return out

class ResidualNetwork(nn.Module):
    def __init__(self,embed_size,dout_p,test = False):
        super().__init__()
        self.layerNorm = nn.LayerNorm(embed_size)
        if test:
            dout_p = 0
        self.dropout = nn.Dropout(dout_p)
    def forward(self,x,sublayer):
        res = self.layerNorm(x)
        res = sublayer(res)
        res= self.dropout(res)
        x = x + res
        return x

class EncodingLayer(nn.Module):
    def __init__(self,embed_size,dout_p,test=False):
        super().__init__()

        self.residualNetwork_1 = ResidualNetwork(embed_size,dout_p=.1,test=test)
        self.residualNetwork_2 = ResidualNetwork(embed_size,dout_p=.1,test=test)

        if test:
            dout_p = 0

        self.attention = Attention(embed_size,dout_p,test=test)
        self.feedForward = nn.Sequential(
            nn.Linear(embed_size,4 * embed_size),
            nn.ReLU(True),
            nn.Dropout(dout_p),
            nn.Linear(4 * embed_size,embed_size)
        )

        if test:
            for name,parameter in self.feedForward.named_parameters():
                parameter.data.fill_(.1)

        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    def forward(self,x,video_mask):
        # does using the same ResNet do any difference?
        x = self.residualNetwork_1(x,lambda x: self.attention(x,x,x,video_mask))
        x = self.residualNetwork_2(x,self.feedForward)
        return x

import unittest
class TestEncodingLayer(unittest.TestCase):
    def test_encoding_layer(self):
        # Create a sample input tensor and video mask
        x = torch.randn(2, 10, 512)
        video_mask = torch.ones(2, 10)
        video_mask[0, 5:] = 0
        video_mask[1, 8:] = 0

        # Initialize the EncodingLayer
        enc = EncodingLayer(512, 0.8)

        # Compute the output of the EncodingLayer
        out = enc(x, video_mask)

        # Check that the output has the expected shape
        self.assertEqual(out.shape, (2, 10, 512))

if __name__ == '__main__':
    unittest.main()
