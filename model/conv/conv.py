import sys
sys.path.append('/home/c1l1mo/projects/VideoAlignment/model')
from transformer.resnet50.resnet50 import ResNet50
import torch 
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self,embed_size,dout,num_context=5):
        super().__init__()
        self.num_context = num_context

        self.resnet = ResNet50(tcc=True)
        self.dropout = nn.Dropout(dout)
        self.conv_layer = nn.Sequential(
            nn.Conv3d(1024,256,kernel_size=3,padding='same'),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256,256,kernel_size=3,padding='same'),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
        )
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dout),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Dropout(dout),
            nn.Linear(256,256),
            nn.ReLU(True),
        )
        self.embedding_layer = nn.Linear(256,embed_size)
    def forward(self,x,video_mask=None,skeleton=None,split=""):
        B , T , C , H , W = x.shape
        x = self.resnet(x)
        x = self.dropout(x)
        x = x.contiguous().view(-1,self.num_context,1024,14,14)
        x = x.permute(0,2,1,3,4)
        x = self.conv_layer(x)
        x = self.maxpool(x)
        x = x.reshape(B , -1 , 256)
        x = self.fc(x)
        x = self.embedding_layer(x)
        return x

import torchvision.models as models

# Embedding Model derived from Tensorflow version
class Embedder(nn.Module):
  def __init__(self, embedding_size,dout, num_context_steps=5):
    super().__init__()

    # Will download pre-trained ResNet50V2 here
    self.resnet = ResNet50(tcc=True)
    self.num_context_steps = num_context_steps
    self.conv_layers = nn.ModuleList([nn.Conv3d(1024, 256, kernel_size=3, padding="same"),nn.Conv3d(256, 256, kernel_size=3, padding="same")])
    self.bn_layers = nn.ModuleList([nn.BatchNorm3d(256)
                                      for _ in range(2)])
    self.maxpool = nn.AdaptiveMaxPool3d(1)
    self.fc_layers = nn.ModuleList([nn.Linear(256, 256)
                                      for _ in range(2)])
    self.embedding_layer = nn.Linear(256, embedding_size)
    self.dropout = nn.Dropout(p=0.1)
  
  def forward(self, frames,video_mask=None,skeleton=None):
    B , T , C , H , W = frames.shape

    x = self.resnet(frames)
    x = x.reshape(-1, self.num_context_steps,1024,14,14)
    x = self.dropout(x)

    x = x.permute(0,2,1,3,4)

    for conv_layer, bn_layer in zip(self.conv_layers,
                                    self.bn_layers):
      x = conv_layer(x)
      x = bn_layer(x)
      x = nn.ReLU(x)

    x = self.maxpool(x)
    x = x.reshape(B, -1, 256)

    for fc_layer in self.fc_layers:
      x = self.dropout(x)
      x = fc_layer(x)
      x = nn.ReLU(x)

    x = self.embedding_layer(x)
    return x

import unittest

class TestConv(unittest.TestCase):
    def test_forward(self):
        # Create a sample input tensor
        B, T , C ,  H, W = 2, 100, 3, 224, 224
        x = torch.randn((B,  T,C, H, W))

        # Create a Conv object with default configuration
        embed_size = 128
        dout = 0.1
        conv = Conv(embed_size, dout)

        # Compute the output for the sample input tensor
        output = conv(x)

        # Check that the output has the correct shape
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (B, 100//5 ,  embed_size))

if __name__ == '__main__':
    unittest.main()