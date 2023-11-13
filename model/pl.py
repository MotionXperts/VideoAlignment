import torch
from torch import nn,utils
import lightning as L
from lightning.pytorch import seed_everything
from transformer.transformer import CARL
import sys
sys.path.append("/home/c1l1mo/projects/VideoAlignment/")
from loss.scl import SCL
import yaml
from easydict import EasyDict as Edict
from dataset.penn_action import PennAction
import os
import random
import numpy as np


with open("/home/c1l1mo/projects/VideoAlignment/result/scl_penn_action/config.yaml", 'r') as config_file:
    config_dict = yaml.safe_load(config_file)
cfg = Edict(config_dict)
cfg.PATH_TO_DATASET = os.path.join("/home/c1l1mo/datasets",cfg.PATH_TO_DATASET)

carl = CARL(cfg)
scl = SCL(cfg)



class LitCARL(L.LightningModule):
    def __init__(self,carl,scl):
        super().__init__()
        self.carl=carl
        self.scl =scl
    def training_step(self,batch,batch_idx):
        original_video,video,label,seq_len,steps,mask,name,skeleton = batch
        batch_size, num_views, num_steps, c, h, w = video.shape
        video = video.view(-1, num_steps, c, h, w)
        
        embs = self.carl(video,video_mask=mask,skeleton=skeleton)
        loss = self.scl.compute_loss(embs,seq_len,steps,mask)
        return loss
    def validation_step(self,batch,batch_id,name):
        original_video,video,label,seq_len,steps,mask,name,skeleton = batch
        batch_size, num_views, num_steps, c, h, w = video.shape
        video = video.view(-1, num_steps, c, h, w)
        embs = self.carl(video,video_mask=mask,skeleton=skeleton)
        loss = self.scl.compute_loss(embs,seq_len,steps,mask)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.MAX_EPOCHS + 1)
        return [optimizer],[scheduler]
    def train_dataloader(self):
        dataset = PennAction(cfg,"train",mode="train",algo="scl")
        data_loader = utils.data.DataLoader(dataset,shuffle=True,batch_size=1,num_workers=cfg.DATA.NUM_WORKERS,pin_memory=True,drop_last=True)
        return data_loader
    def val_dataloader(self):
        eval_dataloader = []
        for dataset_name in cfg.DATASETS:
            eval_dataset = PennAction(cfg,"test",dataset_name=dataset_name,sample_all=False,algo="scl")
            eval_dataloader.append(torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,
                                 num_workers=cfg.DATA.NUM_WORKERS,pin_memory=True,drop_last=True))
        return eval_dataloader
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()
litcarl = LitCARL(carl,scl)

def main():
    seed_everything(7,workers=True)
    trainer = L.Trainer(gpus=[0,1],precision=16,check_val_every_n_epoch=20,max_epochs=cfg.TRAIN.MAX_EPOCHS,sync_batchnorm=True)
    trainer.fit(model=litcarl)

if __name__=="__main__":
    main()