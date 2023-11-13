import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
import traceback
import wandb
import os,sys
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from loss.tcc import TCC
from loss.scl import SCL
import torch.distributed as dist

import utils.dist as du
from utils.config import get_cfg
from utils.parser import parse_args,load_config
from utils.visualize import create_video

from dataset import construct_dataloader
from evaluation.kendalls_tau import KendallsTau
from evaluation.retrieval import Retrieval

from model import save_checkpoint,load_checkpoint,build_model
from torch.utils.tensorboard import SummaryWriter
import logging
import time
from icecream import ic

ic.disable()

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

def train(cfg,algo,model,trainloader,optimizer,scheduler,cur_epoch,summary_writer,DEBUG=False,current_algo=None):
    assert current_algo in ["SCL","TCC"]

    model.train()
    optimizer.zero_grad()
    total_loss = 0
    loss = {}

    if du.is_root_proc():
        trainloader = tqdm(trainloader,total=len(trainloader))
    for cur_iter,(original_video,video,label,seq_len,steps,mask,name,skeleton) in enumerate(trainloader):
<<<<<<< HEAD

=======
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406
        if cur_iter ==0:
            DEBUG = True
        else:
            DEBUG = False
        optimizer.zero_grad()
        scaler = torch.cuda.amp.GradScaler()
        # with torch.cuda.amp.autocast():
        #     loss = algo[current_algo].ori_compute_loss(model,video,seq_len,steps,mask)

        batch_size, num_views, num_steps, c, h, w = video.shape
        video = video.view(-1, num_steps, c, h, w)
        embs = model(video,video_mask=mask,skeleton=skeleton)

        with torch.cuda.amp.autocast():
            loss = algo[current_algo].compute_loss(embs,seq_len.to(embs.device),steps.to(embs.device),mask.to(embs.device),images=original_video,summary_writer=summary_writer,epoch=cur_epoch,split="train")
        ic(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        loss[torch.isnan(loss)] = 0
        ic(loss)
        ic(du.all_reduce([loss])[0].item() / len(trainloader))
        total_loss += du.all_reduce([loss])[0].item() / len(trainloader)


    if du.is_root_proc():
        logger.info(f"epoch: {cur_epoch},total_loss:  {total_loss}")
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
            embs = model(video,skeleton=skeleton,video_mask=mask)
            with torch.cuda.amp.autocast():
                loss = algo[current_algo].compute_loss(embs,seq_len.to(embs.device),steps.to(embs.device),mask.to(embs.device),images=original_video,summary_writer=summary_writer,epoch=cur_epoch,split="train")
            total_loss += du.all_reduce([loss])[0].item() / len(testloader)
        if du.is_root_proc():
            try:
                wandb.log({f"val/loss": total_loss,"custom_step": cur_epoch})
            except:
                pass
            print(f"validation loss: {total_loss:.3f}")

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
                video = video.squeeze(0)
                if cfg.MODEL.EMBEDDER_TYPE != 'conv':
                    assert video.size(1) == frame_label.size(1) == int(seq_len),print(f"video.shape: {video.shape}, frame_label.shape: {frame_label.shape}, seq_len: {seq_len}")
                    with torch.cuda.amp.autocast():
                        emb_feats = model(video,video_mask=None,skeleton=skeleton,split="eval")
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
<<<<<<< HEAD
                        emb_feats = model(input_video,video_mask=None,skeleton=skeleton,split="eval")
=======
                        emb_feats = model(input_video,video_mask=None,skelton=skeleton,split="eval")
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406
                
                
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
                "name":names_list,
                "video":video_list,
                "labels":frame_labels_list,
            }
            KD.evaluate(dataset,epoch,summary_writer,split=split)
            RE.evaluate(dataset,epoch,summary_writer,split=split)
        
            ## check one same q/c pair and check an arbitrary q/c pair
            queries = [4]
            candidates = [5]
            query = np.random.randint(0,len(embs_list))
            candidate = np.random.randint(0,len(embs_list))
            while query == candidate:
                candidate = np.random.randint(0,len(embs_list))
            queries.append(query)
            candidates.append(candidate)

            for query,candidate in zip(queries,candidates):
                video_name = os.path.join(cfg.VISUALIZATION_DIR,f'{split}_{epoch}_{query}_{candidate}.mp4')
                print(f"creating video {video_name}")
                create_video(embs_list[query],video_list[query],embs_list[candidate],video_list[candidate],
                    video_name,use_dtw=True,tsNE_only=(split=="train" or tsNE_only))

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.args = args

    assert "/".join(args.cfg_file.split("/")[:-1]) == cfg.LOGDIR, f"{'/'.join(args.cfg_file.split('/')[:-1])} and {cfg.LOGDIR} does not match, if u want to use ckpt from other directory, comment this line."

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',filename=os.path.join(cfg.LOGDIR,'stdout.log'))

    setup_seed(7)

    dist.init_process_group(backend='nccl', init_method='env://')
    
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.MAX_EPOCHS + 1)
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    if du.is_root_proc() and args.record:
        wandb.init(
            project="VideoAlignment",
            config=cfg,
            group = "DDP",
            name=cfg.LOGDIR.split('/')[-1],
            tags=[cfg.TRAINING_ALGO.split("/")[-1],cfg.PATH_TO_DATASET],
<<<<<<< HEAD
            dir="/home/c1l1mo/tmp/",
=======
            dir="/home/yuansu/tmp/",
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406
            settings=wandb.Settings(_disable_stats=True),
            # id='zxbv9pzm', resume=True, ## resume run
        )

    train_loaders = {}
    train_samplers = {}
    train_eval_loaders = {}
    test_loaders = {}
    test_samplers ={}
    test_eval_loaders = {}

    if cfg.TRAINING_ALGO=='tcc_scl_tcc':
        train_loaders["TCC"],train_samplers["TCC"],train_eval_loaders["TCC"] = construct_dataloader(cfg, 'train',"tcc")
<<<<<<< HEAD
        test_loaders ["TCC"],_                    ,test_eval_loaders ["TCC"] = construct_dataloader(cfg, args.demo_or_inference ,"tcc")
        train_loaders["SCL"],train_samplers["SCL"],train_eval_loaders["SCL"] = construct_dataloader(cfg, 'train',"scl")
        test_loaders ["SCL"],_                    ,test_eval_loaders ["SCL"] = construct_dataloader(cfg, args.demo_or_inference ,"scl")
    else:
        train_loaders[cfg.TRAINING_ALGO.upper()],train_samplers[cfg.TRAINING_ALGO.upper()], train_eval_loaders[cfg.TRAINING_ALGO.upper()] = construct_dataloader(cfg, 'train',cfg.TRAINING_ALGO)
        test_loaders [cfg.TRAINING_ALGO.upper()], _                                       , test_eval_loaders [cfg.TRAINING_ALGO.upper()] = construct_dataloader(cfg, 'test',cfg.TRAINING_ALGO)
=======
        test_loaders ["TCC"],_                    ,test_eval_loaders ["TCC"] = construct_dataloader(cfg, 'val' ,"tcc")
        train_loaders["SCL"],train_samplers["SCL"],train_eval_loaders["SCL"] = construct_dataloader(cfg, 'train',"scl")
        test_loaders ["SCL"],_                    ,test_eval_loaders ["SCL"] = construct_dataloader(cfg, 'val' ,"scl")
    else:
        train_loaders[cfg.TRAINING_ALGO.upper()],train_samplers[cfg.TRAINING_ALGO.upper()], train_eval_loaders[cfg.TRAINING_ALGO.upper()] = construct_dataloader(cfg, 'train',cfg.TRAINING_ALGO)
        test_loaders [cfg.TRAINING_ALGO.upper()], _                                       , test_eval_loaders [cfg.TRAINING_ALGO.upper()] = construct_dataloader(cfg, 'val',cfg.TRAINING_ALGO)
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406


    algo = {"TCC":TCC(cfg),"SCL":SCL(cfg)}
    
    KD = KendallsTau(cfg)
    RE = Retrieval(cfg)

    start_epoch = load_checkpoint(cfg,model,optimizer)

    current_algo = cfg.TRAINING_ALGO.upper()
    try:
        for epoch in range(start_epoch+1,cfg.TRAIN.MAX_EPOCHS+1):
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

            train_sampler.set_epoch(epoch)
            train(cfg,algo,model,train_loader,optimizer,scheduler,epoch,summary_writer,DEBUG=args.debug,current_algo=current_algo)
            if epoch % 100 == 0:
<<<<<<< HEAD
                if du.is_root_proc():
                    if epoch != 0:
                        evaluate(cfg,algo,model,epoch,test_eval_loader,summary_writer,KD,RE,split="test",tsNE_only=True)
            if (epoch+1) % 20 == 0 :
                val(algo,model,test_loader,epoch,summary_writer,current_algo=current_algo) ## validation function should be placed out of du.is_root_proc()
                if du.is_root_proc():
                    save_checkpoint(cfg,model,optimizer,epoch)
=======
                val(algo,model,test_loader,epoch,summary_writer,current_algo=current_algo) ## validation function should be placed out of du.is_root_proc()
                if du.is_root_proc():
                    if epoch != 0:
                        evaluate(cfg,algo,model,epoch,test_eval_loader,summary_writer,KD,RE,split="test",tsNE_only=True)
            if du.is_root_proc() and epoch +1 % 20 ==0:
                save_checkpoint(cfg,model,optimizer,epoch)
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406
            du.synchronize()
    except Exception as e:
        print(traceback.format_exc())
        print(f"{e} occured, saving model before quitting.")
    finally:
<<<<<<< HEAD
        # save_checkpoint(cfg,model,optimizer,epoch)
=======
        save_checkpoint(cfg,model,optimizer,epoch)
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406
        dist.destroy_process_group()

if __name__ == '__main__':
    main()