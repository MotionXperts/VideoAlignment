import torch
import torch.nn as nn
from .positionalEncoding.positionalEncoding import PositionalEncoding
from .encodingLayer.encodingLayer import EncodingLayer
from copy import deepcopy

def clone(module,N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self,cfg,embed_size, final_embed_size,dout_p,test=False):
        super().__init__()
        self.positionalEncoding = PositionalEncoding(None,embed_size,dout_p=0.1,test=test)
        # self.encodingLayers = nn.ModuleList([deepcopy(EncodingLayer(embed_size,dout_p=0.0,test=test)) for _ in range(cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS)])
        self.encodingLayers = clone(EncodingLayer(embed_size,dout_p=0.0,test=test),cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS)
        if test:
            dout_p = 0 
        self.dropout = nn.Dropout(dout_p)
        self.embedding_layer = nn.Linear(embed_size,final_embed_size)
        if test:
            for name,parameter in self.embedding_layer.named_parameters():
                parameter.data.fill_(.1)
    def forward(self,x,video_mask):
        x = self.positionalEncoding(x)
        x_pos = x
        for encodingLayer in self.encodingLayers:
            x = encodingLayer(x,video_mask)
        x_encoding_layer = x

        x = self.embedding_layer(x)

        return x
    
        return x_pos,x_encoding_layer,x
    

import unittest

class TestEncoder(unittest.TestCase):
    def test_encoder(self):
        # Create a sample input tensor and video mask
        x = torch.randn(2, 10, 512)
        video_mask = torch.ones(2, 10)
        video_mask[0, 5:] = 0
        video_mask[1, 8:] = 0

        # Initialize the Encoder
        embed_size = 512
        dout_p = 0.8
        final_embed_size = 256
        encoder = Encoder(embed_size, dout_p, final_embed_size)

        # Compute the output of the Encoder
        out = encoder(x, video_mask)

        # Check that the output has the expected shape
        self.assertEqual(out.shape, (2, 10, final_embed_size))

if __name__ == '__main__':
    unittest.main()
