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
<<<<<<< HEAD
import datetime
=======
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406

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
    else:
        dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
        nns = np.argmin(dists, axis=1)
    return nns

def dist_fn(x, y):
    dist = np.sum((x-y)**2)
    return dist

def viz_align(query_feats, candidate_feats, use_dtw):
    """Align videos based on dynamic time warping."""
    if use_dtw:
        # dtw() returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
        # min_dist represents the similarity of two sequences.
        min_dist, cost_matrix, acc_cost_matrix, path = dtw(query_feats, candidate_feats, dist=dist_fn)
        _, uix = np.unique(path[0], return_index=True) # uix is the index of the unique element
        nns = path[1][uix]
    else:
        nns = []
        for i in range(len(query_feats)):
            nn_frame_id, _ = get_nn( candidate_feats,query_feats[i])
            nns.append(nn_frame_id) 
    return nns

def viz_tSNE(embs,frames,output_path,use_dtw=False,query=0):
    nns = []
    idx = np.arange(len(embs))
    for candidate in range(len(embs)):
        idx[candidate] = candidate
        nns.append(viz_align(embs[query], embs[candidate], use_dtw))
    X = np.empty((0, 128))
    y = []
    frame_idx = []
    for i, video_emb in zip(idx, embs):
        for j in range(len(embs[0])):
            X = np.append(X, np.array([video_emb[nns[i][j]]]), axis=0)
            y.append(int(i))
            frame_idx.append(j)
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
        
    ## print 1D t-SNE
    # X_tsne = manifold.TSNE(n_components=1, init='random', random_state=5, verbose=0).fit_transform(X)
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
    # X_reshape = X_norm.reshape(2,-1)
    # print("X_norm.shape: " , X_reshape.shape)
    # print("x: " , X_reshape[0])
    # print("y: " , X_reshape[1])

    # X_reshape = torch.from_numpy(X_reshape)
    # X_norm = torch.from_numpy(X_norm)
    # ## validate X reshape
    # assert torch.equal(X_reshape.reshape(-1,1),X_norm), f"X_reshape: {X_reshape.reshape(-1,1).shape}, X_norm: {X_norm.shape}"
      
    

    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path)
    plt.close()

def create_video(query_embs, query_frames, key_embs, key_frames, video_path, use_dtw, interval=50, time_stride=1, image_out=False,
    tsNE_only=False):
    """Create aligned videos."""
    nns = align(query_embs, key_embs, use_dtw)
    if time_stride>1:
        query_frames = query_frames[::time_stride]
        nns = nns[::time_stride]
        interval = interval*time_stride
    kendalls_embs = []
    kendalls_embs.append(query_embs)
    kendalls_embs.append(key_embs)
    viz_tSNE(kendalls_embs,None,video_path.split('.mp4')[0]+('.jpg'),use_dtw=use_dtw)
    

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

<<<<<<< HEAD
    
    def update(i):
        """Update plot with next frame."""
        if i ==0:
            start_time = datetime.datetime.now()
            logger.info(f"Start creating video at: {start_time}")
        elif i % len(query_frames)  == 0 :
            logger.info(f"Video created at: {datetime.datetime.now()}, time elapsed: {datetime.datetime.now()-start_time}")
        elif i % 10 == 0:
=======
    def update(i):
        """Update plot with next frame."""
        if i % 10 == 0:
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406
            print(f'{i}/{len(query_frames)}')
        ax[0].imshow(unnorm(query_frames[i]))
        ax[1].imshow(unnorm(key_frames[nns[i]]))
        # Hide grid lines
        ax[0].grid(False)
        ax[1].grid(False)

        # Hide axes ticks
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        plt.tight_layout()
    
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
            frames=np.arange(len(query_frames)),
            interval=interval,
            blit=False)
<<<<<<< HEAD
        
=======
>>>>>>> 47fcb3a6ee4422a4b608b29e8779874a74efa406
        anim.save(video_path, dpi=80)


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
