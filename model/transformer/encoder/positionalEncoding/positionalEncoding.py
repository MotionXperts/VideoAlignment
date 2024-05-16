import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self,cfg,embed_size,dout_p,test=False):
        super().__init__()
        self.embed_size=embed_size  # dimension of the input/output embedding space e.g.: if the input is (T x 256), T is the seqence length and 256 is the embedding space (512)
        if test:
            dout_p = 0
        self.dropout = nn.Dropout(dout_p)  
        self.seq_len = cfg.TRAIN.NUM_FRAMES

    def encode(self,seq_len,embed_size,train_len=None):
        # construct all the odds entries
        odds = np.arange(0,embed_size,2)  ## [0 , 2 , 4 , .... , d_model ]    (if d_model is odd) 
        evens = np.arange(1,embed_size,2) ## [1 , 3 , 5 , .... , d_model-1]   (if d_model is odd)
        
        # construct multiple positional encoding since transformer operates parrellally
        pos_enc_mat = np.zeros((seq_len,embed_size)) ## Shape: (seq_len , d_model)
        
        if train_len is None:
            pos_list = np.arange(seq_len)
        else:
            pos_list = np.linspace(0, train_len-1, num=seq_len)
            


        for i,pos in enumerate((pos_list)):
            pos_enc_mat[i, odds]  = np.sin(pos / (10000 ** (odds / embed_size))) 
            pos_enc_mat[i, evens] = np.cos(pos / (10000 ** (evens / embed_size)))

        return torch.from_numpy(pos_enc_mat).unsqueeze(0) 
    
    def forward(self,x):    
        B, T, embed_size = x.shape
        if T != self.seq_len:
            pos_enc_mat = self.encode(T, embed_size, self.seq_len)
            x = x + pos_enc_mat.type_as(x)
        else:
            pos_enc_mat = self.encode(T, embed_size)
            x = x + pos_enc_mat.type_as(x)
        x = self.dropout(x)
        return x

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PE = PositionalEncoding(None,10,0.8)
    p = PE.encode(seq_len=100,d_model=512)
    p = p.reshape(100,512)
    cax = plt.matshow(p)
    plt.savefig('pe.png')