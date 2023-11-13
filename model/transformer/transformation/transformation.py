import torch
import torch.nn as nn
from icecream import ic
class Transformation(nn.Module):
    def __init__(self,in_channel=2048,test=False):
        super().__init__()
        self.fc_layer = []
        drop_rate = .1
        if test:
            drop_rate = 0
        for layer in range(2):
            self.fc_layer.append(nn.Dropout(drop_rate))
            self.fc_layer.append(nn.Linear(in_channel,512))
            self.fc_layer.append(nn.BatchNorm1d(512))
            self.fc_layer.append(nn.ReLU(True))
            in_channel = 512
        self.fc_layer = nn.Sequential(*self.fc_layer)

        self.video_emb= nn.Linear(512,256)

        if test:
            for name,paramter in self.fc_layer.named_parameters():
                paramter.data.fill_(.1)
            for name,paramter in self.video_emb.named_parameters():
                paramter.data.fill_(.1)
            
    def forward(self,x,B,T):
        x = self.fc_layer(x)
        x_fc = x
        x = self.video_emb(x)
        x_emb = x
        x = x.reshape(B,T,-1)
        return x
    
        # return x_fc,x_emb,x

import unittest

class TestTransformation(unittest.TestCase):
    def test_transformation(self):
        # Create a sample input tensor
        B, T, R = 2, 50, 2048
        x = torch.randn(B, T, R)

        # Initialize the Transformation class
        transformation = Transformation()

        # Compute the output of the Transformation class
        out = transformation(x)

        # Check that the output has the expected shape
        self.assertEqual(out.shape, (B, T, 256))

if __name__ == '__main__':
    unittest.main()