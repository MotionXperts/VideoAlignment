# coding=utf-8
"""Visualize alignment based on nearest neighbor in embedding space."""
import os
import torch
import math
import numpy as np
from scipy.spatial.distance import cdist
import argparse
from utils.dtw import dtw

import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
from sklearn import manifold
import logging
import datetime
from icecream import ic
from functools import partial


logging.getLogger('matplotlib.animation').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

EPSILON = 1e-7

def get_nn(candidate_feats,query_emb):
    # print(candidate_feats.shape , type(candidate_feats) , candidate_feats.dtype)
    # print(np.expand_dims(query_emb,axis=0).shape , type(query_emb), query_emb.dtype)
    query_emb = query_emb.astype(candidate_feats.dtype)
    # dist = cdist(embs, query_emb, axis=1)
    dist = cdist(candidate_feats,np.expand_dims(query_emb,axis=0))
    assert len(dist) == len(candidate_feats)
    return np.argmin(dist), np.min(dist)


def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    query_frame = (query_frame - min_v) / (max_v - min_v)
    return query_frame


def align(query_feats, candidate_feats, use_dtw):
    """Align videos based on nearest neighbor or dynamic time warping."""
    if use_dtw:
        _, _, _, path = dtw(query_feats, candidate_feats, dist='sqeuclidean')
        _, uix = np.unique(path[0], return_index=True)
        nns = path[1][uix]
        ic(path[1],uix,nns)
    else:
        dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
        nns = np.argmin(dists, axis=1)
    return nns

def dist_fn(x, y):
    dist = np.sum((x-y)**2)
    return dist

def viz_tSNE(embs,output_path,use_dtw=True,query=0,labels=None,cfg=None,start_frame=0):
    print(embs[0].shape,embs[1].shape)
    nns = []
    distances = []
    idx = np.arange(len(embs))
    if labels is not None:
        query_valid_frames = np.where(labels[query]>=cfg.EVAL.KENDALLS_TAU_COMPUTE_LABELS)[0]
    else:
        query_valid_frames = np.arange(len(embs[query]))
    for candidate in range(len(embs)):
        idx[candidate] = candidate
        if labels is not None:
            candidates_valid_frames = np.where(labels[candidate]>=cfg.EVAL.KENDALLS_TAU_COMPUTE_LABELS)[0]
        else:
            candidates_valid_frames = np.arange(len(embs[candidate]))
        nn = align(embs[query][query_valid_frames], embs[candidate][candidates_valid_frames], use_dtw)
        nns.append(nn)
        dis = cdist(embs[query][query_valid_frames], embs[candidate][candidates_valid_frames][nn], dist_fn)
        min_index,min_value = np.argmin(dis,axis=1),np.min(dis,axis=1)
        distances.append((min_index,min_value))
    X = np.empty((0, embs[0].shape[1]))
    y = []
    frame_idx = []

    nns[1] = np.unique(nns[1]) ## * because we set np.unique here, the max length of nns[1] will be the same as nns[0] (0,0,0,1,2,3 -> 0,1,2,3)

    for i, video_emb in zip(idx, embs):
        for j in range(len(nns[i])): ## so we can only iterate to the max of nns[i]
            X = np.append(X, np.array([video_emb[nns[i][j]]]), axis=0)
            y.append(int(i))
            frame_idx.append(nns[i][j])
    y = np.array(y)
    frame_idx = np.array(frame_idx)

    #t-SNE
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=0).fit_transform(X)
    plt.figure(figsize=(8, 8))

    #Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize

    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(frame_idx[i]), color=plt.cm.Set1(y[i]), 
                fontdict={'weight': 'bold', 'size': 9})

        # if (i < embs[0].shape[0] and ((i > embs[1].shape[0]+start_frame) or i < start_frame)):
        #     circle = plt.Circle((X_norm[i, 0], X_norm[i, 1]), radius=0.01, color='gray', fill=True)
        # else:
        #     circle = plt.Circle((X_norm[i, 0], X_norm[i, 1]), radius=0.01, color=plt.cm.Set1(y[i]), fill=True)
        # plt.gca().add_patch(circle)

    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path)
    plt.close('all')

def create_video(query_embs, query_frames, key_embs, key_frames, video_path, use_dtw, interval=50, time_stride=1, image_out=False,
    tsNE_only=False,labels=None,cfg=None,stop_frame = None):
    """Create aligned videos."""
    nns = align(query_embs, key_embs, use_dtw)


    if time_stride>1:
        query_frames = query_frames[::time_stride]
        nns = nns[::time_stride]
        interval = interval*time_stride
    ## define the stop frame in the standard video, the moment the player video hit the stop frame, do not continue the video
    if stop_frame is not None:
        ## find the index of the stop frame in the key video
        raw_stop_frame_idx = np.where(nns==stop_frame)
        while len(raw_stop_frame_idx[0]) == 0:
            stop_frame += 1
            raw_stop_frame_idx = np.where(nns==stop_frame)
        stop_frame_idx = raw_stop_frame_idx[0][0]
        # print('stop_frame: ' , stop_frame)
        # print('nn: ' , nns)
        # print('stop_frame_idx: ' , stop_frame_idx)

    kendalls_embs = []
    kendalls_embs.append(query_embs)
    kendalls_embs.append(key_embs)
    viz_tSNE(kendalls_embs,video_path.split('.mp4')[0]+('_tSNE.jpg'),use_dtw=use_dtw,query=0,labels=labels,cfg=cfg)
    # frame_tSNE(kendalls_embs,video_path.split('.mp4')[0]+('.jpg'),use_dtw=use_dtw,labels=labels,cfg=cfg)
    

    plt.figure(figsize=(5,1))
    nns_stride = np.floor(nns/time_stride)
    for t, t_nns in enumerate(nns_stride):
        plt.plot([t, t_nns], [1, 0], 'k--')
        plt.show()
    plt.grid(False)
    plt.savefig(video_path.split('.mp4')[0]+".png")

    if tsNE_only:
        return

    fig, ax = plt.subplots(ncols=2, figsize=(10, 10), tight_layout=True)

    ims = []
    title = fig.suptitle("Initializing Video", fontsize=16)
    def init():
        """Initialize the plot for animation."""
        for i in range(2):
            img_display = ax[i].imshow(unnorm(query_frames[0] if i == 0 else key_frames[nns[0]]))
            ims.append(img_display)
            if labels is not None:
                ax[i].set_title(f"Label: {labels[i][0]}")

            # Hide grid lines and axes ticks
            ax[i].grid(False)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        title = fig.suptitle("Initializing Video", fontsize=16)
        
        return ims,title

    
    def update(i):
        """Update plot with next frame."""
        title.set_text(f'Frame {i}/{len(query_frames)}')

        ax[0].set_title(f"Label: {labels[0][i]}")
        ax[1].set_title(f"Label: {labels[1][nns[i]]}")
        ims[0].set_data(unnorm(query_frames[i]))
        ims[1].set_data(unnorm(key_frames[nns[i]]))

        return ims,title
    
    if image_out:
        image_folder = video_path.split('.mp4')[0]
        os.makedirs(image_folder, exist_ok=True)
        for i in np.arange(len(query_frames)):
            update(i)
            plt.savefig(os.path.join(image_folder, f"frame_{i}.png"))
    else:
        anim = FuncAnimation(
            fig,
            update,
            init_func = init,
            # frames=(len(query_frames)),
            frames = stop_frame_idx if stop_frame is not None else len(query_frames),
            interval=interval,
            blit=False)
        
        anim.save(video_path, dpi=80)
        # logger.info(f"Video saved, time elapsed: {datetime.datetime.now()-start_time}")
        plt.close('all')


def create_multiple_video(query_embs, query_frames, key_embs_list, key_frames_list, video_path, use_dtw, 
                        interval=50):
    """Create aligned videos."""
    K = len(key_embs_list)
    nns_list = []
    for key_embs in key_embs_list:
        nns = align(query_embs, key_embs, use_dtw)
        nns_list.append(nns)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10), tight_layout=True)

    def update(i):
        """Update plot with next frame."""
        if i % 10 == 0:
            print(f'{i}/{len(query_frames)}')
        ax[0, 0].imshow(unnorm(query_frames[i]))
        ax[0, 0].grid(False)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        for k in range(K):
            ax[(k+1)//3,(k+1)%3].imshow(unnorm(key_frames_list[k][nns_list[k][i]]))
            ax[(k+1)//3,(k+1)%3].grid(False)
            ax[(k+1)//3,(k+1)%3].set_xticks([])
            ax[(k+1)//3,(k+1)%3].set_yticks([])
        plt.tight_layout()
    
    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(len(query_frames)),
        interval=interval,
        blit=False)
    anim.save(video_path, dpi=80)


def create_single_video(frames, labels, video_path, interval=50, time_stride=1, image_out=False):
    """Create aligned videos."""
    fig, ax = plt.subplots(ncols=1, figsize=(10, 10), tight_layout=True)
    if time_stride>1:
        frames = frames[::time_stride]
        interval = interval*time_stride

    print(labels[::time_stride])

    

    def update(i):
        """Update plot with next frame."""
        if i % 10 == 0:
            print(f'{i}/{len(frames)}')
        ax.imshow(unnorm(frames[i]))
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    
    if image_out:
        image_folder = video_path.split('.mp4')[0]
        os.makedirs(image_folder, exist_ok=True)
        for i in np.arange(len(frames)):
            update(i)
            plt.savefig(os.path.join(image_folder, f"frame_{i}.png"))
    else:
        anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(len(frames)),
            interval=interval,
            blit=False)
        anim.save(video_path, dpi=80)

def create_simple_video(frames,video_path):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 10), tight_layout=True)
    image = ax.imshow(unnorm(frames[0]))
    def init():
        """Initialize the plot for animation."""
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return image

    def update(i):
        """Update plot with next frame."""
        image.set_data(unnorm(frames[i]))
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return image
    
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=np.arange(len(frames)),
        interval=50,
        blit=False)
    anim.save(video_path, dpi=80)
    plt.show()

def visualize(args, cfg):
    """Visualize alignment."""
    import pickle
    import torch
    from torchvision.io import read_video

    with open(os.path.join(args.data_path, "pouring", 'train.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    for data in dataset:
        name = data["name"]
        video_file = os.path.join(args.data_path, "pouring", data["video_file"])
        if name == args.reference_video:
            print(name)
            video, _, info = read_video(video_file, pts_unit='sec')
            video = video.permute(0,3,1,2).float() / 255.0
            query_frames = video.numpy()
            query_embs = np.arange(len(query_frames)).reshape(-1,1)
        elif name == args.candidate_video:
            print(name)
            video, _, info = read_video(video_file, pts_unit='sec')
            video = video.permute(0,3,1,2).float() / 255.0
            key_frames = video.numpy()
            key_embs = np.arange(len(key_frames)).reshape(-1,1)

    create_video(
        query_embs, query_frames, key_embs, key_frames,
        args.video_path,
        args.use_dtw,
        interval=args.interval)
