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
from utils.visualize import create_video,create_simple_video
from utils.nancy_result import *
from utils.dump_nn_frames import FramesDumper

from dataset import construct_dataloader
from evaluation.kendalls_tau import KendallsTau
from evaluation.retrieval import Retrieval

from model import load_checkpoint,build_model
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

import logging
import torch
from rich.progress import Progress

pylogger = logging.getLogger("torch.distributed")
pylogger.setLevel(logging.ERROR)
logger=logging.getLogger(__name__)

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
    original_video_list = []
    with torch.no_grad():
        for index,dataset in enumerate(loader):
            with Progress() as progress:
                task = progress.add_task(f"Processing {cfg.DATASETS[index]}",total=len(loader[index]))
                for index,(original_video,video,frame_label,seq_len,chosen_steps,mask,names,skeleton) in enumerate(loader[index]):
                    print('evaluating: ' , names)
                    original_video=original_video.squeeze(0).squeeze(0)
                    embs = []
                    seq_len = seq_len.item()
                    assert video.size(0) == 1 # batch_size==1
                    if cfg.MODEL.EMBEDDER_TYPE != 'conv':
                        assert video.size(1) == frame_label.size(1) == int(seq_len),print(f"video.shape: {video.shape}, frame_label.shape: {frame_label.shape}, seq_len: {seq_len}")
                        with torch.cuda.amp.autocast():
                            emb_feats = model(video,video_masks=None,skeleton=skeleton,split="test")
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
                            emb_feats = model(input_video,video_masks=None,split="test")
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
                    original_video_list.append(original_video.squeeze(0))
                    progress.update(task,advance=1)
                dataset = {
                    "embs":embs_list,
                    "names":names_list,
                    "videos":video_list,
                    "labels":frame_labels_list,
                    "original_videos":original_video_list,
                }

            ## append the standard emb to training and validation to generate video
            if standard_entry is not None:
                dataset["embs"].append(standard_entry["embs"])
                dataset["names"].append(standard_entry["name"])
                dataset["videos"].append(standard_entry["video"])
                dataset["labels"].append(standard_entry["labels"])
            
            if len(cfg.DATASETS) > 1:
                dataset["subset_name"] = cfg.DATASETS[index]
            if not no_compute_metrics:
                KD.evaluate(dataset,epoch,summary_writer,split=split)
                RE.evaluate(dataset,epoch,summary_writer,split=split)

            if generate_video:
                queries = [cfg.args.query] if cfg.args.query is not None else []
                candidates = [cfg.args.candidate] if cfg.args.candidate is not None else []
                if cfg.args.random>0:
                    for i in range(cfg.args.random):
                        queries.append(np.random.randint(0,len(names_list)))
                        candidates.append(np.random.randint(0,len(names_list)))
                else:
                    if len(queries)==0:
                        candidate = cfg.args.candidate if cfg.args.candidate is not None else -1
                        for i in range(0,len(dataset["names"])):
                            if names_list[i] == names_list[candidate]:
                                continue
                            queries.append(i)
                            candidates.append(candidate)
                    
                
                for query,candidate in zip(queries,candidates):
                    if du.is_root_proc():
                        if cfg.args.nc :
                            video_name = os.path.join(cfg.LOGDIR,'NC_align',f'{split}_{epoch}_{names_list[query]}_{names_list[candidate]}_({len(embs_list[query])}_{len(embs_list[candidate])}).mp4')
                        else:
                            if cfg.args.group is not None:
                            # if False:
                                os.makedirs(os.path.join(cfg.VISUALIZATION_DIR,cfg.args.group),exist_ok = True)
                                video_name = os.path.join(cfg.VISUALIZATION_DIR,cfg.args.group,f'{split}_{epoch}_{names_list[query]}_{names_list[candidate]}_({len(embs_list[query])}_{len(embs_list[candidate])}).mp4')
                            else:
                                video_name = os.path.join(cfg.VISUALIZATION_DIR,f'{split}_{epoch}_{names_list[query]}_{names_list[candidate]}_({len(embs_list[query])}_{len(embs_list[candidate])}).mp4')

                        # if not os.path.exists(video_name) or cfg.args.tsne:
                        if True:
                            if cfg.args.nc :
                                if not os.path.exists(os.path.join(cfg.LOGDIR,'NC_align')):
                                    os.makedirs(os.path.join(cfg.LOGDIR,'NC_align'),exist_ok = True)
                                align_by_start(embs_list[query],original_video_list[query],embs_list[candidate],original_video_list[candidate],video_name,tsNE_only=cfg.args.tsne)
                            else:
                                labels = np.asarray([frame_labels_list[query],frame_labels_list[candidate]])
                                create_video(embs_list[query],original_video_list[query],embs_list[candidate],original_video_list[candidate],
                                        video_name,use_dtw=("no_dtw" not in video_name),interval=200,labels=labels,cfg=cfg,tsNE_only=cfg.args.tsne,stop_frame=cfg.args.stop_frame)
    ## delete the appended standard as we have done producing video and we want to match the assertion in dump nn frames
    if standard_entry is not None:
        del dataset["embs"][-1]
        del dataset["names"][-1]
        del dataset["videos"][-1]
        del dataset["labels"][-1]
    return dataset

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.args = args
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                            filename=os.path.join(cfg.LOGDIR,'stdout.log'))

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
    elif args.abs_pkl is not None:
        cfg.PATH_TO_DATASET = "/".join(cfg.args.abs_pkl.split('/')[:-1])
        test_name = cfg.args.abs_pkl.split('/')[-1].split('.')[0]
    elif "TEST_NAME" in cfg.DATA:
        test_name = cfg.DATA.TEST_NAME
    else:
        raise Exception("Please specify a test name in config file or in command line.")
    
    train_name = cfg.DATA.TRAIN_NAME
    jump_type = cfg.PATH_TO_DATASET.split('/')[-1]
    if cfg.args.second_align:
        cfg.PATH_TO_DATASET = "/".join(cfg.args.cfg_file.split('/')[:-1])
        train_name = jump_type + "_" +'train_trimmed'
        test_name = jump_type + "_" +'test_trimmed'
        

    
    _,_, test_eval_loader = construct_dataloader(cfg, test_name,cfg.TRAINING_ALGO.split("_")[0],force_test=True)
    algo = {"TCC":TCC(cfg)}    
    KD = KendallsTau(cfg)
    RE = Retrieval(cfg)

    if args.eval_multi:
        from glob import glob
        if du.is_root_proc():
            logger.info("Evaluating multiple checkpoints")
        checkpoints = glob(os.path.join(cfg.LOGDIR,"checkpoints","*.pth"))
        assert args.generate == False, "Cannot generate video when evaluating multiple checkpoints"
    else:
        checkpoints = [args.ckpt]

    for ckpt in checkpoints:
        start_epoch = load_checkpoint(cfg,model,None,ckpt)   
        print("Retreiving test dataset and standard entry ...")
        test_dataset = evaluate(cfg,algo,model,start_epoch,test_eval_loader,summary_writer,KD,RE,split=test_name,generate_video=args.generate,no_compute_metrics=args.no_compute_metrics)
        


    if cfg.args.align_standard or cfg.args.second_align:

        # print("Retreiving train dataset ...")
        # _,_,train_eval_loader = construct_dataloader(cfg, train_name,cfg.TRAINING_ALGO.split("_")[0],force_test=True)
        # train_dataset = evaluate(cfg,algo,model,start_epoch,train_eval_loader,summary_writer,KD,RE,split=cfg.DATA.TRAIN_NAME,generate_video=args.generate,no_compute_metrics=args.no_compute_metrics)

        
        whole_dataset = {
            # "train":train_dataset,
            "test":test_dataset
        }
        standard_entry = {"embs":test_dataset["embs"][0],"name":test_dataset["names"][0],"video":test_dataset["videos"][0],"labels":test_dataset["labels"][0]}
        assert 'standard' in standard_entry['name'] or 'trimmed' in standard_entry['name'] or 'coach' in standard_entry['name'], f"Standard entry name should contain 'standard' or 'trimmed'. {standard_entry['name']}"
        cfg.DATA.TRAIN_NAME = train_name
        cfg.DATA.TEST_NAME = test_name
        FD = FramesDumper(cfg,whole_dataset,standard_entry["embs"],jump_type)
        FD()
    
    dist.destroy_process_group()




if __name__ == '__main__':
    main()