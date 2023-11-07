import torch
import torch.nn as nn

import logging
import sys,os
logger = logging.getLogger(__name__)

class MOCA(nn.Module):
    def __init__(self,cfg=None):
        super().__init__()
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename=os.path.join(cfg.LOGDIR,'stdout.log'))
        self.theta = nn.Conv1d(2048 , 1024 , kernel_size=1 , stride=1 )
        self.phi = nn.Conv1d(2048 , 1024 , kernel_size=1 , stride=1 )
        self.g = nn.Conv1d(2048 , 1024 , kernel_size=1 , stride=1 )
        self.rou = nn.Conv2d(2,1,kernel_size=1,stride=1)
        self.w = nn.Conv1d(1024,2048,kernel_size=1,stride=1)

        self.video_emb = nn.Linear(2048,256)
    def forward(self,x,B,T):
        x = x.view(B,T,2048)
        # B , T , D = x.size()
        ## NSSM
        x_ = x.permute(0,2,1) # B,D,T
        NSSM = x.matmul(x_).softmax(dim=-1) # B,T,T
        logger.info(f"NSSM: {NSSM}")
        ## AttetnionMap

        x = x.permute(0,2,1) # B,D,T
        x_theta = self.theta(x) # B,D/2,T
        x_phi = self.phi(x).permute(0,2,1) # B,T,D/2
        AttentionMap = x_phi.matmul(x_theta).softmax(dim=-1) # B,T,T
        logger.info(f"AttentionMap: {AttentionMap}")
        ## MocaMap
        x_concat = torch.cat([NSSM,AttentionMap],dim=0) # 2B,T,T
        x_concat = x_concat.view(B,-1,T,T) # B,2,T,T
        MocaMap = self.rou(x_concat).squeeze(1).softmax(dim=-1) # B,T,T
        logger.info(f"MocaMap: {MocaMap}")
        ## G branch
        x_g = self.g(x) # B,D/2,T
        Y = x_g.matmul(MocaMap) # B,D/2,T
        logger.info(f"Y: {Y}")
        Wz = self.w(Y)          # B,D,T
        Z = (Wz+x).permute(0,2,1)                # B,T,D
        Z = self.video_emb(Z)                    # B,T,256
        logger.info(f"Z: {Z}")
        return Z


import unittest

class TestMOCA(unittest.TestCase):
    def setUp(self):
        self.model = MOCA()

    def test_forward(self):
        x = torch.randn(2, 10, 2048)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 10, 2048))

if __name__ == '__main__':
    unittest.main()