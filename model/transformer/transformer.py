import torch
import torch.nn as nn
from .encoder.encoder import Encoder
from .resnet50.resnet50 import ResNet50
from .transformation.transformation import Transformation
from .transformation.moca import MOCA
import torch.nn.functional as F

import sys,os
import logging
from icecream import ic

logger = logging.getLogger(__name__)

class CARL(nn.Module):
    def __init__(self,cfg,test=False):
        super().__init__()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename=os.path.join(cfg.LOGDIR,'stdout.log'))
        self.cfg = cfg
        self.resnet50 = ResNet50()

        transfomation_in_channel = 2048
        if hasattr(self.cfg.DATA,'SKELETON') and self.cfg.DATA.SKELETON:
            self.skeleton_linear = nn.Linear(17*3,256)
            transfomation_in_channel += 256
        
        if hasattr(self.cfg.MODEL,'TRANSFORMATION') and self.cfg.MODEL.TRANSFORMATION.TYPE == "MOCA":
            self.transformation = MOCA(cfg)
            print("TRANSFORMATION: MOCA")

        else:
            self.transformation = Transformation(in_channel=transfomation_in_channel,test=test)

        self.encoder = Encoder(cfg,256,cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE,0.1,test=test)
        if self.cfg.MODEL.PROJECTION:
            self.projection = nn.Sequential(
                nn.Linear(cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE,cfg.MODEL.PROJECTION_HIDDEN_SIZE),
                nn.BatchNorm1d(cfg.MODEL.PROJECTION_HIDDEN_SIZE),
                nn.ReLU(True),
                nn.Linear(cfg.MODEL.PROJECTION_HIDDEN_SIZE,cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE)
            )
            if test : 
                for name,parameter in self.projection.named_parameters():
                    parameter.data.fill_(.1)

    def forward(self,x,video_masks=None,skeleton=None,split="train"):
        B , T , C , W ,H = x.shape
        x = self.resnet50(x)

        if hasattr(self.cfg.DATA,'SKELETON') and self.cfg.DATA.SKELETON:
            skeleton = skeleton.view(B*T,17*3)
            skeleton = self.skeleton_linear(skeleton)
            x = torch.cat([x,skeleton],dim=-1)
        
        x = self.transformation(x,B,T)
        x = self.encoder(x,video_masks)
        if self.cfg.MODEL.PROJECTION and split == "train":
            x = x.view(-1,self.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE)
            x = self.projection(x)
            x = x.view(B,T,self.cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE)
            x = F.normalize(x, dim=-1)
        elif self.cfg.MODEL.L2_NORMALIZE:
            x = F.normalize(x, dim=-1)
        return x
    
        # x_res,x_reshape,x_pooling,x_flatten,x = self.resnet50(x)
        # x_fc,x_emb,x = self.transformation(x,B,T)
        # x_transform = x
        # x_pos,x_encoding_layer,x = self.encoder(x,video_mask)
        # return {"x_res":x_res,
        #         "x_reshape":x_reshape,
        #         "x_pooling":x_pooling,
        #         "x_flatten":x_flatten,
        #         "x_fc":x_fc,
        #         "x_emb":x_emb,
        #         "x_transform":x_transform,
        #         "x_pos":x_pos,
        #         "x_encoding_layer":x_encoding_layer,
        #         "x":x}
