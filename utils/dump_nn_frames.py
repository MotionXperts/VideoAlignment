from dtw import *
import torch
import numpy as np
import pickle
import os
import json
from utils.nancy_result import align

class FramesDumper():
    def __init__(self,cfg,dataset,standard_emb,jump_type=None):
        self.cfg = cfg
        self.dataset = dataset
        self.standard_emb = torch.from_numpy(standard_emb)

        ## Make split, input and output based on keys of dataset
        self.split = {}
        self.inputs = []
        self.output_pickles = []

        for split_name in self.dataset.keys():
            self.split[split_name] = []
            if split_name == 'train':
                PKL_NAME = self.cfg.DATA.TRAIN_NAME
            elif split_name == 'test':
                PKL_NAME = self.cfg.DATA.TEST_NAME
            else:
                raise ValueError(f"Unknown split name: {split_name}")
            self.input_pickle = os.path.join(self.cfg.PATH_TO_DATASET,PKL_NAME + '.pkl')
            self.input = self.load_pickle(self.input_pickle)
            self.inputs.append(self.input)
            self.output_pickle = os.path.join(self.cfg.LOGDIR,f'{jump_type}_{split_name}_{cfg.args.second_align}.pkl')
            self.output_pickles.append(self.output_pickle)


        self.split_path = os.path.join(self.cfg.LOGDIR,'splits.json')

    def load_pickle(self,path):
        with open(path,'rb') as f:
            return pickle.load(f)
    
    def optimized_distance_finder(self,self_subtraction_matrix,embs,query_embs):
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
        return min_distance
              
    def find_min_distance_with_standard(self,query_embs):
        tmp = self.standard_emb.expand(self.standard_emb.size(0),self.standard_emb.size(0),-1)
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
        if len(self.standard_emb) < len(query_embs): 
            return 0 
        opt_start_frame = self.optimized_distance_finder(self_subtraction_matrix,self.standard_emb,query_embs)
        print('opt start frame : ',opt_start_frame)
        return opt_start_frame

    def to_torch_tensor(self,data):
        if isinstance(data, np.ndarray):
            try:
                return torch.from_numpy(data)
            except:
                return data
        return data

    def __call__(self):
        for split_name,input_pkl,output_pickle in zip(self.split.keys(),self.inputs,self.output_pickles):
            print(f"Dumping {split_name} ...")
            dataset = self.dataset[split_name]
            
            assert len(dataset["names"]) == len(input_pkl),f"dataset: {len(dataset['names'])}, input_pkl: {len(input_pkl)} is not same"
            for index,entry in enumerate(input_pkl):
                embs = dataset["embs"][index]
                
                ## Need to fix this ...
                # assert abs(len(embs)-len(entry[frame_label_field]))<=2, f"embs: {len(embs)}, frame_label: {len(entry[frame_label_field])} is not same at {entry['name']}"
                
                ## the longer will be key, shorter will be query
                query_embs = embs
                key_embs = self.standard_emb
                if len(query_embs) > len(key_embs):
                    key_embs,query_embs = query_embs,key_embs
                query_embs = self.to_torch_tensor(query_embs)
                key_embs = self.to_torch_tensor(key_embs)    
                
                ## if the start frame exists, that means we are doing the 2nd alignment.
                if 'start_frame' in entry and 'standard' not in entry['video_name']:
                    start_frame = align((query_embs),(key_embs),None)
                    # assert len(key_embs) == len(query_embs),f'key_embs: {len(key_embs)}, query_embs: {len(query_embs)} in {entry["name"]}'
                    subtraction = key_embs[start_frame:start_frame+len(query_embs)] - query_embs
                    entry["subtraction"] = subtraction
                    entry['annotations_label'] = entry['original_annotations_label'][entry['start_frame']:entry['start_frame']+len(entry['subtraction'])]
                else: ## otherwise, do the 1st alignment. Which find the start frame of the longer video, subtraction for the 1st alignment is meaningless.
                    start_frame = align((query_embs),(key_embs),None)
                    end_frame = start_frame + len(query_embs)
                    entry['start_frame'] = start_frame
                    entry['end_frame'] = end_frame
                    ## record if std is longer or not
                    entry['standard_longer'] = len(self.standard_emb) > len(embs)
                    if "GT" in self.cfg.DATA.TRAIN_NAME:
                        entry['subtraction'] = key_embs[start_frame:end_frame] - query_embs

                self.split[split_name].append(entry["name"])

            if self.cfg.args.overwrite:
                with open(output_pickle,'wb') as f:
                    pickle.dump(input_pkl,f)
                print(f"Dumped {split_name} to {output_pickle}")

        with open(self.split_path,'w') as f:
            json.dump(self.split,f)
                