import os,sys
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import pickle
import torch
import random
import numpy as np
from loss.tcc import TCC
import torch.distributed as dist

import utils.dist as du
from utils.parser import parse_args,load_config
from utils.visualize import create_video
from utils.nancy_result import *
from utils.dump_nn_frames import FramesDumper

from dataset import construct_dataloader
from evaluation.kendalls_tau import KendallsTau
from evaluation.retrieval import Retrieval

from model import load_checkpoint,build_model
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

import logging

pylogger = logging.getLogger("torch.distributed")
pylogger.setLevel(logging.ERROR)

ic.disable()

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

def evaluate(cfg,algo,model,epoch,loader,summary_writer,KD,RE,split="val",generate_video=True,no_compute_metrics=False,standard_entry = None):
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
        for index,dataset in enumerate(loader):
            for cur_iter,(original_video,video,frame_label,seq_len,chosen_steps,mask,names,skeleton) in enumerate(loader[index]):
                video = video.squeeze(0)
                original_video=original_video.squeeze(0)
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
                    original_video=original_video.squeeze(0)
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

            ## append the standard emb to training and validation to generate video
            if standard_entry is not None:
                dataset["embs"].append(standard_entry["embs"])
                dataset["name"].append(standard_entry["name"])
                dataset["video"].append(standard_entry["video"])
                dataset["labels"].append(standard_entry["labels"])
            
            if not no_compute_metrics:
                KD.evaluate(dataset,epoch,summary_writer,split=split)
                RE.evaluate(dataset,epoch,summary_writer,split=split)

            

            if generate_video:
                if cfg.args.query is not None and cfg.args.candidate is not None:
                    queries = [cfg.args.query]
                    candidates = [cfg.args.candidate]
                elif cfg.args.random>0:
                    queries = []
                    candidates = []
                    for i in range(cfg.args.random):
                        queries.append(np.random.randint(0,len(names_list)))
                        candidates.append(np.random.randint(0,len(names_list)))
                else:
                    queries = [-1]
                    candidates = []
                    for i in range(0,len(dataset["name"])):
                        queries.append(-1)
                        candidates.append(i)
                
                for query,candidate in zip(queries,candidates):
                    while candidate == query:
                        candidate = np.random.randint(0,len(names_list))
                    if  du.is_root_proc():
                        if cfg.args.nc :
                            video_name = os.path.join(cfg.LOGDIR,'NC_align',f'{split}_{epoch}_{names_list[query]}_{names_list[candidate]}_({len(embs_list[query])}_{len(embs_list[candidate])}).mp4')
                        else:
                            video_name = os.path.join(cfg.VISUALIZATION_DIR,f'{split}_{epoch}_{names_list[query]}_{names_list[candidate]}_({len(embs_list[query])}_{len(embs_list[candidate])}).mp4')
                        print(f"generating video {video_name}")

                        if not os.path.exists(video_name) and "cam2_GX010274" not in video_name:
                            if cfg.args.nc :
                                align_by_start(cfg,video_name,dataset,query,candidate)
                            else:
                                labels = np.asarray([frame_labels_list[query],frame_labels_list[candidate]])
                                create_video(embs_list[query],video_list[query],embs_list[candidate],video_list[candidate],
                                        video_name,use_dtw=("no_dtw" not in video_name),interval=200,labels=labels,cfg=cfg)

    ## delete the appended standard as we have done producing video and we want to match the assertion in dump nn frames
    if standard_entry is not None:
        del dataset["embs"][-1]
        del dataset["name"][-1]
        del dataset["video"][-1]
        del dataset["labels"][-1]
    return dataset
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

    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    if args.demo_or_inference is not None:
        test_name = args.demo_or_inference
    elif "TEST_NAME" in cfg.DATA:
        test_name = cfg.DATA.TEST_NAME
    else:
        raise Exception("Please specify a test name in config file or in command line.")


    testloader,_, test_eval_loader = construct_dataloader(cfg, test_name,cfg.TRAINING_ALGO.split("_")[0],force_test=True)
    algo = {"TCC":TCC(cfg)}    
    KD = KendallsTau(cfg)
    RE = Retrieval(cfg)
    start_epoch = load_checkpoint(cfg,model,None,args.ckpt)

    
    print("Retreiving test dataset and standard entry ...")
    test_dataset = evaluate(cfg,algo,model,start_epoch,test_eval_loader,summary_writer,KD,RE,split=test_name,generate_video=args.generate,no_compute_metrics=args.no_compute_metrics)
    standard_entry = {"embs":test_dataset["embs"][-1],"name":test_dataset["name"][-1],"video":test_dataset["video"][-1],"labels":test_dataset["labels"][-1]}


    if cfg.args.align_standard:
        with open(os.path.join(cfg.PATH_TO_DATASET,test_name+".pkl"),'rb') as f:
            standard_assertion = pickle.load(f)
        assert "standard" in standard_assertion[-1]["name"]

        _,_,train_eval_loader = construct_dataloader(cfg, "long_train_label",cfg.TRAINING_ALGO.split("_")[0])
        _,_,val_eval_loader = construct_dataloader(cfg, "long_val_label",cfg.TRAINING_ALGO.split("_")[0])

        print("Retreiving train dataset ...")
        train_dataset = evaluate(cfg,algo,model,start_epoch,train_eval_loader,summary_writer,KD,RE,split="long_train_label",generate_video=args.generate,no_compute_metrics=args.no_compute_metrics,standard_entry = standard_entry)
        print("Retreiving test dataset ...")
        val_dataset = evaluate(cfg,algo,model,start_epoch,val_eval_loader,summary_writer,KD,RE,split="long_val_label",generate_video=args.generate,no_compute_metrics=args.no_compute_metrics,standard_entry = standard_entry)
        
        whole_dataset = {
            "train":train_dataset,
            "val":val_dataset,
            "test":test_dataset
        }

        FD = FramesDumper(cfg,whole_dataset,standard_entry["embs"])
        FD()
    
    dist.destroy_process_group()




if __name__ == '__main__':
    main()