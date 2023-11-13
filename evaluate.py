import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import os,sys
import math
import torch
import random
import numpy as np
from loss.tcc import TCC
from loss.carl_tcc import TCC as CARL_TCC
import torch.distributed as dist

import utils.dist as du
from utils.parser import parse_args,load_config
from utils.visualize import create_video

from dataset import construct_dataloader
from evaluation.kendalls_tau import KendallsTau
from evaluation.retrieval import Retrieval

from model import load_checkpoint,build_model
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

def evaluate(cfg,algo,model,epoch,loader,summary_writer,KD,RE,split="val",generate_video=True):
    embs_list = []
    steps_list = []
    seq_lens_list = []
    frame_labels_list = []
    names_list = []
    input_lens_list = []

    max_frames_per_batch = cfg.EVAL.FRAMES_PER_BATCH
    num_contexts = cfg.DATA.NUM_CONTEXTS

    model.eval()
    embs_list = []
    video_list = []
    with torch.no_grad():
        for index,dataset in enumerate(loader):
            for cur_iter,(original_video,video,frame_label,seq_len,chosen_steps,mask,names,skeleton) in enumerate(loader[index]):
                video = video.squeeze(0)
                embs = []
                seq_len = seq_len.item()
                assert video.size(0) == 1 # batch_size==1
                if cfg.MODEL.EMBEDDER_TYPE != 'conv':
                    assert video.size(1) == frame_label.size(1) == int(seq_len),print(f"video.shape: {video.shape}, frame_label.shape: {frame_label.shape}, seq_len: {seq_len}")
                    with torch.cuda.amp.autocast():
                        emb_feats = model(video,video_mask=None,skeleton=skeleton)
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
                        emb_feats = model(input_video,video_mask=None)
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

<<<<<<< HEAD
            queries = []
            candidates = []

            for _ in range(15):
=======
            queries = [3]
            candidates = [8]

            for _ in range(5):
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406
                queries.append(np.random.randint(0,len(names_list)))
                candidates.append(np.random.randint(0,len(names_list)))
            # queries=  [23,25,28,4,4]
            # candidates = [8,26,14,25,16]

            for query,candidate in zip(queries,candidates):
                while candidate == query:
                    candidate = np.random.randint(0,len(names_list))
                if generate_video and du.is_root_proc():
                    video_name = os.path.join(cfg.VISUALIZATION_DIR,f'{split}_{epoch}_{names_list[query]}_{names_list[candidate]}_({len(embs_list[query])}_{len(embs_list[candidate])}).mp4')
                    print(f"generate video {video_name}")
                    if not os.path.exists(video_name):
                        create_video(embs_list[query],video_list[query],embs_list[candidate],video_list[candidate],
                                video_name,use_dtw=("no_dtw" not in video_name),interval=200)
                    # video_name = os.path.join(cfg.VISUALIZATION_DIR,f'{split}_{epoch}_{names_list[query]}_{names_list[candidate]}_no_dtw.mp4')
                    # print(f"generate video {video_name}")
                    # if not os.path.exists(video_name):
                    #     create_video(embs_list[query],video_list[query],embs_list[candidate],video_list[candidate],
                    #             video_name,use_dtw=("no_dtw" not in video_name),interval=200)

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.args = args

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
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

<<<<<<< HEAD
    trainloader,train_sampler, train_eval_loader = construct_dataloader(cfg, 'train',cfg.TRAINING_ALGO.split("_")[0])
    testloader,_, test_eval_loader = construct_dataloader(cfg, args.demo_or_inference,cfg.TRAINING_ALGO.split("_")[0])
=======
    trainloader,train_sampler, train_eval_loader = construct_dataloader(cfg, 'train')
    testloader,_, test_eval_loader = construct_dataloader(cfg, args.demo_or_inference)
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406

    algo = {"TCC":TCC(cfg),"CARL_TCC":CARL_TCC(cfg)}    
    KD = KendallsTau(cfg)
    RE = Retrieval(cfg)

    start_epoch = load_checkpoint(cfg,model,optimizer,args.ckpt)

    # align_by_start(cfg,model,start_epoch,test_eval_loader)
    # evaluate(cfg,algo,model,start_epoch,train_eval_loader,summary_writer,KD,split="train")
    evaluate(cfg,algo,model,start_epoch,test_eval_loader,summary_writer,KD,RE,split=args.demo_or_inference,generate_video=args.generate)
    dist.destroy_process_group()

if __name__ == '__main__':
    main()