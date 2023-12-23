from transformers import CvtModel,CvtConfig
from torch import nn
import torch
from icecream import ic

class ConvolutionalTransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        configuration = CvtConfig(cls_token=[False,False,False],depth=[1,4,16])
        cvt = CvtModel.from_pretrained("microsoft/cvt-21",config=configuration)
        if self.cfg.MODEL.ENCODER.FINETUNE_LAST:
            self.cvt = cvt
            self.finetune = (list((list(cvt.children())[0]).stages[2].children())[1])
            assert len(self.finetune) == 16,len(self.finetune)
        else:
            self.cvt = cvt
            # self.finetune = nn.Identity()
        for param in self.cvt.parameters():
            param.requires_grad = False
        for param in self.finetune.parameters():
            param.requires_grad = True
    def forward(self,x):
        B , T , C , W ,H = x.shape
        x = x.contiguous().view(-1,C,W,H)
        self.cvt.eval()
            
        if self.cfg.MODEL.ENCODER.FINETUNE_LAST:
            with torch.no_grad():
                hidden_state,cls_token = (list(self.cvt.children())[0]).stages[0](x)
                hidden_state,cls_token = (list(self.cvt.children())[0]).stages[1](hidden_state)
                hidden_state = (list((list(self.cvt.children())[0]).stages[2].children())[0])(hidden_state)
                batch_size, num_channels, height, width = hidden_state.shape
                # rearrange b c h w -> b (h w) c"
                hidden_state = hidden_state.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            for layer in self.finetune:
                hidden_state = layer(hidden_state,height,width)
            hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
            return hidden_state
        else:
            with torch.no_grad():
                x = self.cvt(x)
            return x.last_hidden_state