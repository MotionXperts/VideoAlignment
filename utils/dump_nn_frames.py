from dtw import *
import torch
import numpy as np
import pickle
import os
import json
from icecream import ic
from utils.nancy_result import align

class FramesDumper():
    def __init__(self,cfg,dataset,standard_emb):
        self.cfg = cfg
        self.dataset = dataset
        self.standard_emb = torch.from_numpy(standard_emb)
        self.input_train_pickle = os.path.join(self.cfg.PATH_TO_DATASET,self.cfg.DATA.TRAIN_NAME + '.pkl')
        # self.input_val_pickle = os.path.join(self.cfg.PATH_TO_DATASET,'long_val_label.pkl')
        self.input_test_pickle = os.path.join(self.cfg.PATH_TO_DATASET,self.cfg.DATA.TEST_NAME+ '.pkl')

        self.input_train = self.load_pickle(self.input_train_pickle)
        # self.input_val = self.load_pickle(self.input_val_pickle)
        self.input_test = self.load_pickle(self.input_test_pickle)
        self.original_test_length = len(self.input_test)

        self.output_train_pickle = os.path.join(self.cfg.LOGDIR,'output_new_trimmed_train_label.pkl')
        # self.output_val_pickle = os.path.join(self.cfg.LOGDIR,'output_val_label.pkl')
        self.output_test_pickle = os.path.join(self.cfg.LOGDIR,'output_new_trimmed_test_label.pkl')
        # self.standard_output_pickle = os.path.join(self.cfg.LOGDIR,'standard_label.pkl')

        self.split_path = os.path.join(self.cfg.LOGDIR,'splits.json')
        self.split = {'train':[],'val':[],'test':[]}

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


    def __call__(self):
        frame_label_field = "frame_label"
        if self.cfg.args.long:
            frame_label_field = "original_frame_label"

        # for split_name,split,output_pickle in zip(['train','val','test'],[self.input_train,self.input_val,self.input_test],[self.output_train_pickle,self.output_val_pickle,self.output_test_pickle]):
        for split_name,split,output_pickle in zip(['train','test'],[self.input_train,self.input_test],[self.output_train_pickle,self.output_test_pickle]):
            print(f"Dumping {split_name} ...")
            dataset = self.dataset[split_name]

            # this code is adjusted as "before dumping, seprate standard from test set"
            # if split_name == "test": # because we dont want to make standard's label, delete it here.
            #     del dataset["embs"][-1]
            #     del dataset["name"][-1]
            #     del dataset["video"][-1]
            #     del dataset["labels"][-1]
            assert len(dataset["names"]) == len(split),ic(len(dataset["names"]),len(split))
            for index,entry in enumerate(split):
                embs = dataset["embs"][index]
                # if "locate_module_label" not in entry:
                #     entry["locate_module_label"] = torch.zeros_like(entry["frame_label"])
                # labels = entry["locate_module_label"]
                # assert len(labels) == len(entry["frame_label"])
                assert len(embs) == len(entry[frame_label_field]), f"embs: {len(embs)}, frame_label: {len(entry[frame_label_field])} is not same at {entry['name']}"
                
                ## the longer will be key, shorter will be query
                query_embs = embs
                key_embs = self.standard_emb
                if len(query_embs) > len(key_embs):
                    key_embs,query_embs = query_embs,key_embs
                try:
                    query_embs = torch.from_numpy(query_embs)
                except:
                    pass
                try:
                    key_embs = torch.from_numpy(key_embs)
                except:
                    pass

                if 'standard' not in entry['video_name']:
                    entry['annotations_label'] = entry['original_annotations_label'][entry['start_frame']:entry['start_frame']+len(entry['subtraction'])]
                    
                start_frame = align((query_embs),(key_embs),None)


                end_frame = start_frame + len(query_embs)
                # assert start_frame<= len(entry["frame_label"]) and end_frame <= len(entry["frame_label"]) ## will fail in slack cases
                # entry["locate_module_start_frame"] = start_frame
                # entry["locate_module_end_frame"] = end_frame
                # entry["locate_module_label"] = labels[start_frame:end_frame]

                # aligned_embs = embs[start_frame:end_frame]
                subtraction = key_embs[start_frame:end_frame] - query_embs
                # assert aligned_embs.shape == self.standard_emb.shape,ic(embs.shape,self.standard_emb.shape)
                entry["subtraction"] = subtraction
                
                entry['start_frame'] = start_frame

                self.split[split_name].append(entry["name"])

            if self.cfg.args.overwrite:
                # if split_name == "test":
                #     print(f"Dumping standard ...")
                #     standard_split = [split[-1]]
                #     split = split[:-1]
                #     assert "standard" in standard_split[0]["name"]
                #     with open(self.standard_output_pickle,'wb') as f:
                #         pickle.dump(standard_split,f)
                #     print(f"Dumped standard to {self.standard_output_pickle}")
                #     assert (len(split) + len(standard_split)) == self.original_test_length, ic(len(split),len(standard_split),self.original_test_length)
                with open(output_pickle,'wb') as f:
                    pickle.dump(split,f)
                print(f"Dumped {split_name} to {output_pickle}")

        with open(self.split_path,'w') as f:
            json.dump(self.split,f)
                