import os,sys
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
import traceback
import wandb
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from loss.tcc import TCC
from loss.scl import SCL
from loss.lav import LAV
import torch.distributed as dist

import utils.dist as du
from utils.config import get_cfg
from utils.parser import parse_args,load_config
from utils.visualize import create_video
from utils.dtw import dtw

from dataset import construct_dataloader
from evaluation.kendalls_tau import KendallsTau
from evaluation.retrieval import Retrieval

from model import save_checkpoint,load_checkpoint,build_model
from torch.utils.tensorboard import SummaryWriter
import logging
from icecream import ic
import sys

USER = os.environ["USER"]

ic.disable()

pylogger = logging.getLogger("torch.distributed")
pylogger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

def get_lr(optimizer):
    return [param_group["lr"] for param_group in optimizer.param_groups]

def train(cfg,algo,model,trainloader,optimizer,scheduler,cur_epoch,summary_writer,scaler,DEBUG=False,current_algo=None,lav=None,KD=None):
    assert current_algo in ["SCL","TCC"]

    model.train()
    optimizer.zero_grad()
    total_loss = 0

    # DistributedSampler shuffle based on epoch and seed
    if hasattr(trainloader.sampler, 'set_epoch'):
        trainloader.sampler.set_epoch(cur_epoch)
    if hasattr(trainloader.batch_sampler, 'set_epoch'):
        trainloader.batch_sampler.set_epoch(cur_epoch)

    loss = {}

    if du.is_root_proc():
        trainloader = tqdm(trainloader,total=len(trainloader))
    for cur_iter,(original_video,video,label,seq_len,steps,mask,name,skeleton) in enumerate(trainloader):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if cfg.TRAINING_ALGO == "scl":
                batch_size, num_views, num_steps, c, h, w = video.shape
            else:
                num_views, num_steps, c, h, w = video.shape
            video = video.view(-1, num_steps, c, h, w)
            embs = model(video,video_masks=mask,skeleton=skeleton)
            loss = algo[current_algo].compute_loss(embs,seq_len.to(embs.device),steps.to(embs.device),mask.to(embs.device),DEBUG=False,images=original_video,summary_writer=summary_writer,epoch=cur_epoch,split="train")
            if lav is not None:
                lav_loss = lav.compute_loss(embs,steps.to(embs.device),seq_len.to(embs.device))
                loss=loss+lav_loss
        scaler.scale(loss).backward()

        if cfg.OPTIMIZER.GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
        scaler.step(optimizer)
        
        scaler.update()
        
        loss[torch.isnan(loss)] = 0
        total_loss += du.all_reduce([loss])[0].item() / len(trainloader)
        # logger.info(f"embs: {embs.sum()}")
        # logger.info(f"total_loss: {total_loss}")

    if du.is_root_proc():
        if DEBUG:
            ## original video
            # summary_writer.add_video(f'train/ori_video', video, cur_epoch, fps=1)
            ## add cost matrix as image
            _, _, _, path = dtw(embs[0].detach().cpu(), embs[1].detach().cpu(), dist='sqeuclidean')
            _, uix = np.unique(path[0], return_index=True)
            nns = path[1][uix]
            
            ## aligned video
            print("shape of video[0]: ", video[0].shape)
            print("shape of video[1]: ", video[1].shape)
            print("uix: " , uix)
            print("nns: ", nns)
            aligned_video = torch.stack([video[0], video[1][nns]])
            print("shape of aligned video: ", aligned_video.shape)
            summary_writer.add_video(f'train/aligned_video', aligned_video, cur_epoch, fps=1)


        logger.info("epoch {}, train loss: {:.3f}".format(cur_epoch, total_loss))
        ## add 2(batch size) per-frame videos to tensorboard
        summary_writer.add_scalar('train/loss', total_loss, cur_epoch)
        summary_writer.add_scalar('train/learning_rate', get_lr(optimizer)[0], cur_epoch)
        
        try:
            wandb.log({f"train/loss": total_loss,"custom_step": cur_epoch})
            wandb.log({"learning_rate": get_lr(optimizer)[0],"custom_step": cur_epoch})
        except:
            pass
    

    if cur_epoch != cfg.TRAIN.MAX_EPOCHS-1:
        scheduler.step()

def val(algo,model,testloader,cur_epoch,summary_writer,current_algo=None):
    model.eval()
    with torch.no_grad():
        loss = 0
        total_loss = 0
        if du.is_root_proc():
            testloader = tqdm(testloader,total=len(testloader))
        for cur_iter,(original_video,video,label,seq_len,steps,mask,name,skeleton) in enumerate(testloader):
            batch_size, num_views, num_steps, c, h, w = video.shape
            video = video.view(-1, num_steps, c, h, w)
            embs = model(video,skeleton=None,video_masks=mask.to(video.device))
            with torch.cuda.amp.autocast():
                loss = algo[current_algo].compute_loss(embs,seq_len.to(embs.device),steps.to(embs.device),mask.to(embs.device),images=original_video,summary_writer=summary_writer,epoch=cur_epoch,split="train")
            total_loss += du.all_reduce([loss])[0].item() / len(testloader)
        if du.is_root_proc():
            try:
                wandb.log({f"val/loss": total_loss,"custom_step": cur_epoch})
            except:
                pass

            logger.info(f"epoch: {cur_epoch}, validation loss: {total_loss:.3f}")

def evaluate(cfg,algo,model,epoch,loader,summary_writer,KD,RE,split="val",tsNE_only=True):
    embs_list = []
    steps_list = []
    seq_lens_list = []
    frame_labels_list = []
    names_list = []
    input_lens_list = []

    model.eval()
    embs_list = []
    video_list = []
    with torch.no_grad():
        for index,_ in enumerate(loader):
            if du.is_root_proc():
                sub_loader = tqdm(loader[index],total=len(loader[index]))
            for cur_iter,(original_video,video,frame_label,seq_len,chosen_steps,mask,names,skeleton) in enumerate(sub_loader):
                embs = []
                assert video.size(0) == 1 # batch_size==1
                try: seq_len = seq_len.item()
                except: seq_len=seq_len
                if cfg.MODEL.EMBEDDER_TYPE != 'conv':
                    assert video.size(1) == frame_label.size(1) == int(seq_len),print(f"video.shape: {video.shape}, frame_label.shape: {frame_label.shape}, seq_len: {seq_len}")
                    with torch.cuda.amp.autocast():
                        emb_feats = model.module(video.to(model.device),video_masks=None,skeleton=skeleton,split="eval")
                else:
                    
                    assert video.size(1) == frame_label.size(1) == int(seq_len),print(f"video.shape: {video.shape}, frame_label.shape: {frame_label.shape}, seq_len: {seq_len}")
                    steps = torch.arange(0, seq_len, cfg.DATA.SAMPLE_ALL_STRIDE)
                    context_stride = cfg.DATA.CONTEXT_STRIDE
                    steps = steps.view(-1,1) + context_stride*torch.arange(-(cfg.DATA.NUM_CONTEXTS-1), 1).view(1,-1)
                    steps = torch.clamp(steps.view(-1), 0, seq_len - 1)
                    # Select data based on steps
                    video = video.squeeze(0)
                    input_video = video[steps.long()]
                    input_video = input_video.unsqueeze(0)
                    with torch.cuda.amp.autocast():
                        emb_feats = model.module(input_video.to(model.device),video_masks=None,skeleton=skeleton,split="eval")
                
                
                embs.append(emb_feats[0].cpu())
                valid = (frame_label[0]>=0)
                embs = torch.cat(embs, dim=0)
                embs_list.append(embs.numpy())
                frame_labels_list.append(frame_label[0][valid].cpu().numpy())
                seq_lens_list.append(seq_len)
                input_lens_list.append(len(video[0]))
                steps_list.append(chosen_steps[0].cpu().numpy())
                names_list.append(names[0])
                video_list.append(video.squeeze(0).permute(0,2,3,1))
            dataset = {
                "embs":embs_list,
                "names":names_list,
                "videos":video_list,
                "labels":frame_labels_list,
            }
            if len(cfg.DATASETS) > 1:
                dataset["subset_name"] = cfg.DATASETS[index]
            KD.evaluate(dataset,epoch,summary_writer,split=split)
            RE.evaluate(dataset,epoch,summary_writer,split=split)


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for n, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if 'backbone' in n and cfg.MODEL.TRAIN_BASE != 'train_all':
                if cfg.MODEL.TRAIN_BASE == 'frozen':
                    continue
                elif cfg.MODEL.TRAIN_BASE == 'only_bn':
                    if is_bn:
                        bn_params.append(p)
            else:
                if is_bn:
                    bn_params.append(p)
                else:
                    non_bn_parameters.append(p)

    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    optim_params = [
        {"params": bn_params, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY},
    ]

    if cfg.OPTIMIZER.TYPE == "MomentumOptimizer":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            momentum=0.9,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.TYPE == "AdamOptimizer":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER.TYPE == "AdamWOptimizer":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.OPTIMIZER.TYPE)
        )

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.args = args

    assert "/".join(args.cfg_file.split("/")[:-1]) == cfg.LOGDIR, f"{'/'.join(args.cfg_file.split('/')[:-1])} and {cfg.LOGDIR} does not match, if u want to use ckpt from other directory, comment this line."

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',filename=os.path.join(cfg.LOGDIR,'stdout.log'))

    dist.init_process_group(backend='nccl', init_method='env://')

    setup_seed(7)    
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], 
            output_device=args.local_rank,find_unused_parameters=False)

    optimizer = construct_optimizer(model,cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.MAX_EPOCHS + 1)
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    if du.is_root_proc() and args.record:
        wandb.init(
            project="VideoAlignment",
            config=cfg,
            group = "DDP",
            name=cfg.LOGDIR.split('/')[-1],
            tags=[cfg.TRAINING_ALGO.split("/")[-1],cfg.PATH_TO_DATASET],
            dir=f"/home/${USER}/tmp/",
            settings=wandb.Settings(_disable_stats=True),
            # id='zxbv9pzm', resume=True, ## resume run
        )

    train_loaders = {}
    train_samplers = {}
    train_eval_loaders = {}
    test_loaders = {}
    test_samplers ={}
    test_eval_loaders = {}

    if hasattr(cfg.DATA,'TRAIN_NAME'):
        train_name = cfg.DATA.TRAIN_NAME
    else:
        train_name = 'train'

    if hasattr(cfg.DATA,'TEST_NAME'):
        test_name = cfg.DATA.TEST_NAME
    elif args.demo_or_inference is not None:
        test_name = args.demo_or_inference
    else:
        test_name = 'processed_videos_test'
    

    if cfg.TRAINING_ALGO=='tcc_scl_tcc':
        train_loaders["TCC"],train_samplers["TCC"],train_eval_loaders["TCC"] = construct_dataloader(cfg, train_name,"tcc")
        test_loaders ["TCC"],_                    ,test_eval_loaders ["TCC"] = construct_dataloader(cfg, args.demo_or_inference ,"tcc")
        train_loaders["SCL"],train_samplers["SCL"],train_eval_loaders["SCL"] = construct_dataloader(cfg, train_name,"scl")
        test_loaders ["SCL"],_                    ,test_eval_loaders ["SCL"] = construct_dataloader(cfg, args.demo_or_inference ,"scl")
    else:
        train_loaders[cfg.TRAINING_ALGO.upper()],train_samplers[cfg.TRAINING_ALGO.upper()], train_eval_loaders[cfg.TRAINING_ALGO.upper()] = construct_dataloader(cfg, train_name,cfg.TRAINING_ALGO)
        test_loaders [cfg.TRAINING_ALGO.upper()], _                                       , test_eval_loaders [cfg.TRAINING_ALGO.upper()] = construct_dataloader(cfg, test_name,cfg.TRAINING_ALGO)

    
    algo = {"TCC":TCC(cfg),"SCL":SCL(cfg)}
    lav = None
    if 'LAV' in cfg:
        lav = LAV(cfg)
    
    KD = KendallsTau(cfg)
    RE = Retrieval(cfg)

    start_epoch = load_checkpoint(cfg,model,optimizer)
    scaler = torch.cuda.amp.GradScaler()
    current_algo = cfg.TRAINING_ALGO.upper()

    try:
        for epoch in range(start_epoch,cfg.TRAIN.MAX_EPOCHS+2):
            if "_" in cfg.TRAINING_ALGO:
                if epoch < cfg.TRAIN.FIRST_STAGE_EPOCHS:
                    current_algo = cfg.TRAINING_ALGO.split("_")[0].upper()
                elif epoch < cfg.TRAIN.SECOND_STAGE_EPOCHS:
                    current_algo = cfg.TRAINING_ALGO.split("_")[1].upper()
                else:
                    current_algo = cfg.TRAINING_ALGO.split("_")[2].upper()
            train_loader = train_loaders[current_algo]
            train_sampler = train_samplers[current_algo]
            test_loader = test_loaders[current_algo]
            test_eval_loader = test_eval_loaders[current_algo]
            
            train(cfg,algo,model,train_loader,optimizer,scheduler,epoch,summary_writer,scaler,DEBUG=args.debug,current_algo=current_algo,lav=lav,KD=KD)
            if (epoch+1) % 50 == 0 :
                val(algo,model,test_loader,epoch,summary_writer,current_algo=current_algo) ## validation function should be placed out of du.is_root_proc()
                if du.is_root_proc():
                    save_checkpoint(cfg,model,optimizer,epoch)
                    evaluate(cfg,algo,model,epoch,test_eval_loader,summary_writer,KD,RE,split="test")
            du.synchronize()
    except Exception as e:
        print(traceback.format_exc())
        print(f"{e} occured, saving model before quitting.")
    finally:
        if epoch != start_epoch:
            save_checkpoint(cfg,model,optimizer,epoch)
        dist.destroy_process_group()

if __name__ == '__main__':
    main()