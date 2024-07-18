from .visualize import viz_tSNE
from dtw import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import numpy as np


def unnorm(query_frame):
  min_v = query_frame.min()
  max_v = query_frame.max()
  query_frame = (query_frame - min_v) / (max_v - min_v)
  return query_frame

def optimized_distance_finder(self_subtraction_matrix,embs,query_embs):
    """
    Credit to Jason :D
    Algorithm: 
        A : [a0, a1, a2, a3, a4 ... ... an]
        B : [b0, b1, b2, b3, b4 ... bm] (n > m)
        Compute distance B and sliding window(A)
        np.sum([a0-b0, a1-b1, a2-b2, a3-b3, a4-b4 ... am-bm]) -> dist0
        np.sum([a1-b0, a2-b1, a3-b2, a4-b3, a5-b4 ... a(m+1)-bm]) -> dist1
        np.sum([a2-b0, a3-b1, a4-b2, a5-b3, a6-b4 ... a(m+2)-bm]) -> dist2
        ...
        a1-b0 is equal to a1-a0 + a0-b0, a2-b1 is equal to a2-a1 + a1-b1, a3-b2 is equal to a3-a2 + a2-b2 ...
        Similarly, 
        a2-b0 is equal to a2-a0 + a0-b0, a3-b1 is equal to a3-a1 + a1-b1, a4-b2 is equal to a4-a2 + a2-b2 ...
    So we only need to compute dist0 and a(0~m) - a(0~m)
    Note: This cannot work on l2 distance because square is incoporated, so we use abs to calcualate distances
    """
    ### Compute dist0
    n = self_subtraction_matrix.size(0)
    m = len(query_embs)
    
    ## dist.shape = m x query_embs.size(-1)
    dist0 = embs[:m] - query_embs ## dont do abs and sum here because we need the signed value to do algorithm
    
    distances = [torch.sum((dist0)**2)]
    for i in range(1, n-m+1):
        ## index diagonallly in self_subtraction_matrix
        distances.append(torch.sum((dist0 + (torch.diagonal(self_subtraction_matrix,i).transpose(0,1)[:m]) )**2))

    min_distance = torch.argmin(torch.stack(distances))
    x = np.arange(0, n-m+1)
    plt.plot(x, distances, '-ro', markevery=[min_distance])
    plt.savefig('a_min_dist.png')
    plt.clf()
    return min_distance

def align(query_embs,key_embs,name) -> (int):
    """
    Compute which time window of key_embs is most similar to the query_embs(user's input)
    inputs:
    @ query_embs: Tu , 512
    @ key_embs: Ts , 512
    """    
    tmp = key_embs.expand(key_embs.size(0),key_embs.size(0),-1)
    self_subtraction_matrix = (tmp - tmp.transpose(0,1))
    """
          a0    a1   a2  a3     a4 ...
    a0  a0-a0 a1-a0 a2-a0 a3-a0 a4-a0
    a1  a0-a1 a1-a1 a2-a1 a3-a1 a4-a1
    a2  a0-a2 a1-a2 a2-a2 a3-a2 a4-a2
    a3  a0-a3 a1-a3 a2-a3 a3-a3 a4-a3
    a4  a0-a4 a1-a4 a2-a4 a3-a4 a4-a4
    ...
    """

    if len(key_embs) < len(query_embs): 
        return 0
    opt_start_frame = optimized_distance_finder(self_subtraction_matrix,key_embs,query_embs)
    print('opt start frame : ',opt_start_frame)
    return opt_start_frame

def align_by_start(query_embs,query_frames,key_embs,key_frames,output_name,tsNE_only=False):

    ## the longer will be key, shorter will be query
    if len(query_embs) > len(key_embs):
      key_embs,query_embs = query_embs,key_embs
      key_frames,query_frames = query_frames,key_frames
    start_frame = align(torch.from_numpy(query_embs),torch.from_numpy(key_embs),None)
    start_frames = [0,start_frame]
    frames = [query_frames,key_frames[start_frame:start_frame+len(query_frames)]]
    # embs = [query_embs,key_embs[start_frame:start_frame+len(query_embs)]] ## why do this? i forget
    embs = [key_embs,query_embs]
    viz_tSNE(embs,output_name.replace('.mp4','.png'))
    if tsNE_only:
      return
    gen_result(start_frames,frames,output_name)

def gen_result(start_frames, frames,output_name):
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
  
  num_total_frames = min([len(frames[0]), len(frames[1])])
  def update(i):
    ims[0].set_data(unnorm(frames[0][i]))
    ax[0].set_title('START {} '.format(start_frames[0]), fontsize = 14)
    ims[1].set_data(unnorm(frames[1][i]))
    ax[1].set_title('START {}'.format(start_frames[1]), fontsize = 14)
      

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