# coding=utf-8
"""Evaluate embeddings on downstream tasks."""

import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import wandb
import os
import math
import torch
import pprint
import numpy as np
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

import utils.dist as du
from utils.parser import parse_args, load_config #setup_train_dir
from model import build_model, save_checkpoint, load_checkpoint
from utils.visualize import create_video
from dataset import construct_dataloader

TRAIN_VIS = False
ALIGN_STANDARD = False
logger = logging.getLogger(__name__)

def get_embeddings_dataset(cfg, model, data_loader):
    """Get embeddings from a one epoch iterator."""
    max_frames_per_batch = cfg.EVAL.FRAMES_PER_BATCH
    num_contexts = cfg.DATA.NUM_CONTEXTS
    embs_list = []
    steps_list = []
    seq_lens_list = []
    frame_labels_list = []
    names_list = []
    input_lens_list = []

    model.eval()
    with torch.no_grad():
        for video, frame_label, seq_len, chosen_steps, video_masks, names in data_loader:
            assert video.size(0) == 1 # batch_size==1
            assert video.size(1) == frame_label.size(1) == int(seq_len.item())
            embs = []
            seq_len = seq_len.item()
            num_batches = int(math.ceil(float(seq_len)/max_frames_per_batch))
            frames_per_batch = int(math.ceil(float(seq_len)/num_batches))
            for i in range(num_batches):
                curr_idx = i * frames_per_batch
                num_steps = min(seq_len - curr_idx, frames_per_batch)
                steps = torch.arange(curr_idx, curr_idx+num_steps)
                if num_contexts != 1:
                    # Get multiple context steps depending on config at selected steps.
                    context_stride = cfg.DATA.CONTEXT_STRIDE
                    steps = steps.view(-1,1) + context_stride*torch.arange(-(num_contexts-1), 1).view(1,-1)
                steps = torch.clamp(steps.view(-1), 0, seq_len - 1)
                curr_data = video[:, steps]
                # print(i, num_steps, seq_len, curr_data.shape)
                if cfg.USE_AMP:
                    with torch.cuda.amp.autocast():
                        emb_feats = model(curr_data, num_steps)
                else:
                    emb_feats = model(curr_data, num_steps)
                embs.append(emb_feats[0].cpu())
            valid = (frame_label[0]>=0)
            embs = torch.cat(embs, dim=0)
            embs_list.append(embs[valid].numpy())
            frame_labels_list.append(frame_label[0][valid].cpu().numpy())
            seq_lens_list.append(seq_len)
            input_lens_list.append(len(video[0]))
            steps_list.append(chosen_steps[0].cpu().numpy())
            names_list.append(names[0])

        dataset = {'embs': embs_list,
                    'labels': frame_labels_list,
                    'seq_lens': seq_lens_list,
                    'input_lens': input_lens_list,
                    'steps': steps_list,
                    'names': names_list}

        logger.info(f"embeddings_dataset size: {len(dataset['embs'])}")
    return dataset

def evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                    iterator_tasks, embedding_tasks, cur_epoch, summary_writer,generate_video=True):
    """Evaluate learnt embeddings on downstream tasks."""

    metrics = {}

    if True:
        for i, dataset_name in enumerate(cfg.DATASETS):
            dataset = {'name': dataset_name}
            ## computes scores
            dataset['val_dataset'] = get_embeddings_dataset(cfg, model, val_emb_loader)
            # for task_name, task in embedding_tasks.items():
            #     if task_name not in metrics:
            #         metrics[task_name] = {}
            #     try:
            #         metrics[task_name][dataset_name] = task.evaluate(dataset, cur_epoch, summary_writer)
            #     except Exception as e:
            #         logger.error(f"{dataset['name']} of {task_name} failed. due to {e}")
            if not generate_video:
                continue
            if TRAIN_VIS :
                logger.info(f"generating training embeddings for {dataset_name} dataset at {cur_epoch}.")
                dataset['train_dataset'] = get_embeddings_dataset(cfg, model, train_emb_loader[i])
                print("generating visualization for train alignment")
                time_stride=2
                K = 6
                q_id = 0
                k_ids:list = [40,50,60,70,80,90]
                query_data = dataset['train_dataset']['embs'][q_id]
                key_data_list = [dataset['train_dataset']['embs'][k_id] for k_id in k_ids]

                key_frames_list = [0 for _ in range(K)]
                for data_id, data in enumerate(train_emb_loader[i].dataset.dataset):
                    if data['name'] == dataset['train_dataset']['names'][q_id]:
                        query_video = train_emb_loader[i].dataset[data_id][0].permute(0,2,3,1)
                    else:
                        print(dataset['train_dataset']['names'])
                        for k, k_id in enumerate(k_ids):
                            if data['name'] == dataset['train_dataset']['names'][k_id]:
                                key_frames_list[k] = train_emb_loader[i].dataset[data_id][0].permute(0,2,3,1)
                key_video, key_data = key_frames_list[0], key_data_list[0]
                create_video(query_data, query_video, key_data, key_video, 
                        os.path.join(cfg.LOGDIR, f'alignment_{cur_epoch}_train.mp4'), use_dtw=True, interval=50, time_stride=time_stride, image_out=False)

            else:
                logger.info(f"generating val embeddings for {dataset_name} dataset at {cur_epoch}.")
                logger.info("generating visualization for video alignment")
                time_stride=1
                if not ALIGN_STANDARD:
                    q_id = 4
                    # K = len(dataset['val_dataset']['names'])
                    K = cfg.args.K
                    k_ids:list = [25]
                    query_data = dataset['val_dataset']['embs'][q_id]
                    key_data_list = [dataset['val_dataset']['embs'][k_id] for k_id in k_ids]
                    query_name = dataset['val_dataset']['names'][q_id]
                    key_frames_list = [0 for _ in range(K)]
                    for data_id, data in enumerate(val_emb_loader.dataset.dataset):
                        if data['name'] == dataset['val_dataset']['names'][q_id]:
                            query_video = val_emb_loader.dataset[data_id][0].permute(0,2,3,1)
                        else:
                            for k, k_id in enumerate(k_ids):
                                if data['name'] == dataset['val_dataset']['names'][k_id]:
                                    key_frames_list[k] = val_emb_loader.dataset[data_id][0].permute(0,2,3,1)
                    for k,k_id in enumerate((k_ids)):
                        key_video, key_data = key_frames_list[k], key_data_list[k]
                        video_name = dataset['val_dataset']['names'][k_id]
                        if query_name == video_name:
                            continue
                        logger.info(f"generating video for {query_name} and {video_name}")

                        video_name = os.path.join(cfg.LOGDIR, 'visualization', f'alignment_{cur_epoch}_{query_name}_{video_name}.mp4')
                        if not os.path.exists(video_name):
                            create_video(query_data, query_video, key_data, key_video, 
                                    video_name, use_dtw=True, interval=200, time_stride=time_stride, image_out=False)
                else:
                    for index,name in enumerate(dataset['val_dataset']['names']):
                        print(index, ": " , name)
                    # k_id = int(input("input key id: "))
                    k_id = 0
                    Q = len(dataset['val_dataset']['names'])
                    q_ids:list = [i for i in range(1,Q)]
                    key_data = dataset['val_dataset']['embs'][k_id]
                    print(f"q_ids: {q_ids}")
                    query_data_list = [dataset['val_dataset']['embs'][q_id] for q_id in q_ids ]
                    key_name = dataset['val_dataset']['names'][k_id]
                    query_frames_list = [0 for _ in range(Q)]
                    for data_id, data in enumerate(val_emb_loader.dataset.dataset):
                        if data['name'] == dataset['val_dataset']['names'][k_id]:
                            key_video = val_emb_loader.dataset[data_id][0].permute(0,2,3,1)
                        else:
                            for q, q_id in enumerate(q_ids):
                                if data['name'] == dataset['val_dataset']['names'][q_id]:
                                    query_frames_list[q] = val_emb_loader.dataset[data_id][0].permute(0,2,3,1)
                    for q in range(len(q_ids)):
                        query_video, query_data = query_frames_list[q], query_data_list[q]
                        query_name = dataset['val_dataset']['names'][q]
                        create_video(query_data, query_video, key_data, key_video, 
                                os.path.join(cfg.LOGDIR, f'visualization/alignment_{cur_epoch}_{query_name}_{key_name}_.mp4'), use_dtw=True, interval=50, time_stride=time_stride, image_out=False)
                
            del dataset
    
def evaluate():
    """Evaluate embeddings."""
    args = parse_args()
    cfg = load_config(args)
    os.makedirs(os.path.join(cfg.LOGDIR,"visualization"),exist_ok=True)
    # setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.args = args

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    print("ARGS RECORD IS:" , args.record)
    if du.is_root_proc() and args.record:
        wandb.init(
            project="carl",
            config=cfg,
            group="evaluation",
            name=cfg.LOGDIR.split('/')[-1],
            tags=[cfg.TRAINING_ALGO.split("/")[-1],cfg.PATH_TO_DATASET,args.demo_or_inference],
            dir="/home/c1l1mo/tmp/wandb",
            settings=wandb.Settings(_disable_stats=True)
        )

    # Setup summary writer.
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'eval_logs'))

    # Build the video model
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], 
            output_device = args.local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.OPTIMIZER.LR.INITIAL_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,)
    start_epoch = load_checkpoint(cfg, model, optimizer)

    # Setup Dataset Iterators from train and val datasets.
    # train_loader, train_emb_loader = construct_dataloader(cfg, "train")
    train_loader, train_emb_loader = None,None
    print(args.demo_or_inference)
    val_loader,_, val_emb_loader = construct_dataloader(cfg, args.demo_or_inference)

    evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                        None, None, start_epoch, summary_writer)
    
    wandb.finish()

if __name__ == '__main__':
    evaluate()
