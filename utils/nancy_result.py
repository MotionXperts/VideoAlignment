import datetime
from dtw import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import torch
import numpy as np
from icecream import ic


def unnorm(query_frame):
  min_v = query_frame.min()
  max_v = query_frame.max()
  query_frame = (query_frame - min_v) / (max_v - min_v)
  return query_frame

def align_by_start(cfg,video_name,dataset,query,candidate):
    ## in this case, load 2 video and compare their min distance to the standard embedding
    # std_loader, std_eval_loader = construct_dataloader(cfg, 'standard')

    embs_list = dataset["embs"]
    video_list = dataset["video"]
    names_list = dataset["name"]


    assert 'standard' in names_list[-1],ic(names_list[-1])
    std_emb = embs_list[-1] ## the standard_0 locates at the end of the "Skating" dataset.
    

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
            min_dist, cost_matrix, acc_cost_matrix, path = dtw(query_embs, standard_emb, dist=dist_fn) ## * the dtw yields
            min_dists.append(min_dist)
        if len(emb) == len(standard_emb):
           start_frame = 0
        else:
          start_frame = min_dists.index(min(min_dists)) ## * the smallest value (set it as start frame)
      

        # Plot min_dist calculation
        # os.makedirs(os.path.join(CONFIG.LOGDIR,'NC_align'),exist_ok = True)
        # x = np.arange(0, len(emb)-len(standard_emb))
        # plt.plot(x, min_dists, '-ro', markevery=[start_frame])
        # plt.savefig(os.path.join(CONFIG.LOGDIR,'NC_align' , f'min_dists_{emb_name}'))
        # plt.clf()
        return start_frame
    
    query_start_frame = get_start_frame(cfg, embs_list[query], names_list[query], std_emb)
    candidate_start_frame = get_start_frame(cfg, embs_list[candidate], names_list[candidate], std_emb)

    ic(f"{query_start_frame}/{len(embs_list[query])} {candidate_start_frame}/{len(embs_list[candidate])}")
    start_frame = [query_start_frame, candidate_start_frame]
    frames = [video_list[query][query_start_frame:query_start_frame+len(std_emb)], video_list[candidate][candidate_start_frame:candidate_start_frame+len(std_emb)]]
    query,candidate = 0,1

    gen_result(start_frame,frames, video_name,query,candidate)

def gen_result(start_frames, frames,output_name,query,candi ):
  # Create subplots
  nrows = len(frames)
  fig, ax = plt.subplots(ncols=nrows,figsize=(10, 10),tight_layout=True)
  
  ims = []
  def init():
    for k in range(nrows):
      ims.append(ax[k].imshow(unnorm(frames[k][0])))
      ax[k].grid(False)
      ax[k].set_xticks([])
      ax[k].set_yticks([])
    return ims
  
  num_total_frames = min([len(frames[query]), len(frames[candi])])
  def update(i):
    if i % 10 ==0:
      print(f'{i}/{num_total_frames}')
    ims[0].set_data(unnorm(frames[query][i]))
    ax[0].set_title('START {} '.format(start_frames[query]), fontsize = 14)
    ims[1].set_data(unnorm(frames[candi][i]))
    ax[1].set_title('START {}'.format(start_frames[candi]), fontsize = 14)
      

  # # The one with larger start_frame needs to be played first
  # first = query if (start_frames[query] > start_frames[candi]) else candi
  # second = candi if (first==query) else query


  # # The second video starts playing at start_frame
  # start_frame = start_frames[first] - start_frames[second]
  # if (start_frames[first] + len(frames[second])) > len(frames[first]):
  #   num_total_frames = start_frames[first] + len(frames[second])
  # else:
  #   num_total_frames = len(frames[first])

  # def update(i):
  #   if i % 10 == 0:
  #     print(f'{i}/{num_total_frames}')

  #   if i < len(frames[first]):
  #     ims[first].set_data(unnorm(frames[first][i]))
  #   else:
  #     ims[first].set_data(unnorm(frames[first][-1]))
  #   ax[first].set_title('START {} '.format(start_frames[first]), fontsize = 14)
  #   ax[second].set_title('START {}'.format(start_frames[second]), fontsize = 14)

  #   if i >= start_frame and i < (start_frame+len(frames[second])):
  #     ims[second].set_data(unnorm(frames[second][i-start_frame]))
  #   elif i < start_frame:
  #     ims[second].set_data(unnorm(frames[second][0]))
  #   else:
  #     ims[second].set_data(unnorm(frames[second][-1]))
  #   plt.tight_layout()
  #   return ims


  # Create animation
  anim = FuncAnimation(
      fig,
      update,
      init_func=init,
      frames=np.arange(num_total_frames),
      interval=300,
      blit=False)
  anim.save(output_name, dpi=80)

  plt.close('all')