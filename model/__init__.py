import os
import torch
from natsort import natsorted
from .transformer.transformer import CARL 
from .conv.conv import Conv
from .carl_transformer.transformer import TransformerModel as CARL_Transformer
from icecream import ic
from utils import dist as du

def save_checkpoint(cfg, model, optimizer, epoch):
    path = os.path.join(cfg.LOGDIR, "checkpoints")
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, "checkpoint_epoch_{:05d}.pth".format(epoch))
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    }
    if not os.path.exists(ckpt_path):
        torch.save(checkpoint, ckpt_path)
        print(f"Saving epoch {epoch} checkpoint at {ckpt_path}")

def load_checkpoint(cfg,model,optimizer,name=None):
    checkpoint_dir = os.path.join(cfg.LOGDIR, "checkpoints")
    if os.path.exists(checkpoint_dir) and cfg.args.ckpt != "no":
        checkpoints = os.listdir(checkpoint_dir)
        if len(checkpoints) > 0:
            ## sort the files in checkpoint dir
            if name is not None:
                checkpoint_path = name
            else:
                checkpoint_path = natsorted(checkpoints)[-1]
            checkpoint = torch.load(os.path.join(checkpoint_dir,checkpoint_path))
            model.module.load_state_dict(checkpoint["model_state"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if du.is_root_proc():
                print(f"LOADING CHECKPOINT AT {checkpoint_path}")
            return checkpoint["epoch"] 
    if du.is_root_proc():
        print("checkpoint not found")
    return 0 

def build_model(cfg):
    if cfg.args.carl:
        return CARL_Transformer(cfg,test=False)
    elif cfg.MODEL.EMBEDDER_TYPE=='transformer':
        print("BUILDING TRANSFORMER")
        return CARL(cfg)
    else:
        print("BUILDING CONV")
        return Conv(cfg,cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE,cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE,cfg.DATA.NUM_CONTEXTS)