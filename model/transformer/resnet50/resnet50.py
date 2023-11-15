import torch
import torch.nn as nn
import torchvision.models as models
import math
from icecream import ic

class ResNet50(nn.Module):
    def __init__(self,tcc=False):
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.tcc = tcc
        self.backbone = nn.Sequential(*list(model.children())[:-3])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.finetune = list(model.children())[-3]
        self.pooling = nn.AdaptiveMaxPool2d(1)
    def forward(self,x):
        B , T , C , W ,H = x.shape
        frames_per_batch = 40
        num_blocks = int(math.ceil(float(T)/frames_per_batch))
        output = []
        for i in range(num_blocks):
            if (i+1) * frames_per_batch > T:
                processing = x[:,i*frames_per_batch:]
            else:
                processing = x[:,i*frames_per_batch:(i+1)*frames_per_batch]
            processing = processing.contiguous().view(-1,C,W,H)
            ## feed into resnet
            self.backbone.eval()
            with torch.no_grad():
                processing = self.backbone(processing)
            if not self.tcc:
                processing = self.finetune(processing)
                processing= processing.view(B,-1,2048,7,7)
            else: 
                processing = processing.view(B,-1,1024,14,14)
            output.append(processing)
        x = torch.cat(output,dim=1)
        x_res = x
        if not self.tcc:
            x = x.view(B*T,2048,7,7)
            x_reshape = x
            x = self.pooling(x)
            x_pooling=x
            x = x.flatten(start_dim=1)
            x_flatten=x

        return x
    
        return x_res,x_reshape,x_pooling,x_flatten,x


import unittest

class TestResNet50(unittest.TestCase):
    def test_resnet50(self):
        # Create a sample input tensor
        B, T, C, W, H = 2, 40, 3, 224, 224
        x = torch.randn(B, T, C, W, H)

        # Initialize the ResNet50 module
        resnet50 = ResNet50()

        # Compute the output of the ResNet50 module
        out = resnet50(x)

        # Check that the output has the expected shape
        self.assertEqual(out.shape, (B, T , 2048))

if __name__ == '__main__':
    unittest.main()
