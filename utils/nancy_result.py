import datetime
from dtw import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import torch
import numpy as np

def get_embedding(model,loader):
    steps_list = []
    seq_lens_list = []
    frame_labels_list = []
    names_list = []
    input_lens_list = []

    model.eval()
    embs_list = []
    video_list = []
    with torch.no_grad():
        for cur_iter,(video,frame_label,seq_len,chosen_steps,mask,name) in enumerate(loader):
            assert video.size(0) == 1 # batch_size==1
            assert video.size(1) == frame_label.size(1) == int(seq_len.item())
            embs = []
            seq_len = seq_len.item()
            with torch.cuda.amp.autocast():
                emb_feats = model(video,video_mask=None)
            embs.append(emb_feats[0].cpu())

            valid = (frame_label[0]>=0)
            embs = torch.cat(embs, dim=0)
            embs_list.append(embs[valid].numpy())
            frame_labels_list.append(frame_label[0][valid].cpu().numpy())
            seq_lens_list.append(seq_len)
            input_lens_list.append(len(video[0]))
            steps_list.append(chosen_steps[0].cpu().numpy())
            names_list.append(name)
            video_list.append(video.squeeze(0).permute(0,2,3,1))
    dataset = {
        "embs":embs_list,
        "name":names_list,
        "video":video_list,
        "labels":frame_labels_list,
    }
    return dataset

def align_by_start(cfg,model,epoch,loader):
    ## in this case, load 2 video and compare their min distance to the standard embedding
    std_loader, std_eval_loader = construct_dataloader(cfg, 'standard')
    model.eval()
    with torch.no_grad():
        for cur_iter,(video,label,seq_len,steps,mask,name) in enumerate(std_eval_loader):
            embs = model(video,mask)
            std_emb = embs[0]
    dataset = get_embedding(model,loader)
    embs_list = dataset["embs"]
    video_list = dataset["video"]
    names_list = dataset["name"]

    query = 0 
    candidate = 1
    def dist_fn(x, y):
      dist = np.sum((x-y)**2)
      return dist

    def get_start_frame(CONFIG, emb, emb_name, standard_emb):
        min_dists = []

        if len(emb) < len(standard_emb): ## pad the emb so that it can at least match with the first frame
            for i in range(len(standard_emb)-len(emb)+1):
                emb=  np.vstack((emb,(emb[-1])))

        for i in range(len(emb)-len(standard_emb)): ## * use sliding window
            query_embs = emb[i:i+len(standard_emb)]   ## * compare in which window
            min_dist, cost_matrix, acc_cost_matrix, path = dtw(query_embs, standard_emb.cpu().numpy(), dist=dist_fn) ## * the dtw yields
            min_dists.append(min_dist)
        
        start_frame = min_dists.index(min(min_dists)) ## * the smallest value (set it as start frame)

        # Plot min_dist calculation
        os.makedirs(os.path.join(CONFIG.LOGDIR,'visualization'),exist_ok = True)
        x = np.arange(0, len(emb)-len(standard_emb))
        plt.plot(x, min_dists, '-ro', markevery=[start_frame])
        plt.savefig(os.path.join(CONFIG.LOGDIR,'visualization' , f'min_dists_{emb_name}'))
        plt.clf()
        return start_frame
    
    query_start_frame = get_start_frame(cfg, embs_list[query], names_list[query], std_emb)
    candidate_start_frame = get_start_frame(cfg, embs_list[candidate], names_list[candidate], std_emb)

    start_frame = [query_start_frame, candidate_start_frame]

    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_result(cfg,start_frame, [video_list[query],video_list[candidate]], names_list[query], now_time)

def gen_result(CONFIG,start_frames, frames,output_name, now_time, query=0, candi=1 ):
  OUTPUT_PATH = os.path.join(CONFIG.LOGDIR,'visualization',f'align_{output_name}_{format(now_time)}.mp4')
  # Create subplots
  nrows = len(frames)
  fig, ax = plt.subplots(
        ncols=nrows,
        figsize=(10 * nrows, 10 * nrows),
        tight_layout=True)
  
  def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    query_frame = (query_frame - min_v) / (max_v - min_v)
    return query_frame

  ims = []
  def init():
    k = 0
    for k in range(nrows):
      ims.append(ax[k].imshow(
          unnorm(frames[k][0])))
      ax[k].grid(False)
      ax[k].set_xticks([])
      ax[k].set_yticks([])
    return ims

  ims = init()

  # The one with larger start_frame needs to be played first
  first = query if (start_frames[query] > start_frames[candi]) else candi
  second = candi if (first==query) else query
  # The second video starts playing at start_frame
  start_frame = start_frames[first] - start_frames[second]
  if (start_frames[first] + len(frames[second])) > len(frames[first]):
    num_total_frames = start_frames[first] + len(frames[second])
  else:
    num_total_frames = len(frames[first])

  def update(i):
    if i < len(frames[first]):
      ims[first].set_data(unnorm(frames[first][i]))
    else:
      ims[first].set_data(unnorm(frames[first][-1]))
    ax[first].set_title('FRAME {}'.format(i), fontsize = 14)

    if i >= start_frame and i < (start_frame+len(frames[second])):
      ims[second].set_data(unnorm(frames[second][i-start_frame]))
    elif i < start_frame:
      ims[second].set_data(unnorm(frames[second][0]))
    else:
      ims[second].set_data(unnorm(frames[second][-1]))
    plt.tight_layout()
    return ims

  # Create animation
  anim = FuncAnimation(
      fig,
      update,
      frames=np.arange(num_total_frames),
      interval=100,
      blit=False)
  anim.save(OUTPUT_PATH, dpi=40)

  plt.close()