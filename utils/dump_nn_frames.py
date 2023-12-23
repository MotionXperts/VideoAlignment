from dtw import *
import torch
import numpy as np
import pickle
import os
from icecream import ic

class FramesDumper():
    def __init__(self,cfg,dataset,standard_emb):
        self.cfg = cfg
        self.dataset = dataset
        self.standard_emb = standard_emb
        self.input_train_pickle = os.path.join(self.cfg.PATH_TO_DATASET,'long_train_label.pkl')
        self.input_val_pickle = os.path.join(self.cfg.PATH_TO_DATASET,'long_val_label.pkl')
        self.input_test_pickle = os.path.join(self.cfg.PATH_TO_DATASET,'long_test_with_assumed_standard_label.pkl')

        self.input_train = self.load_pickle(self.input_train_pickle)
        self.input_val = self.load_pickle(self.input_val_pickle)
        self.input_test = self.load_pickle(self.input_test_pickle)
        self.original_test_length = len(self.input_test)

        self.output_train_pickle = os.path.join(self.cfg.LOGDIR,'output_train.pkl')
        self.output_val_pickle = os.path.join(self.cfg.LOGDIR,'output_val.pkl')
        self.output_test_pickle = os.path.join(self.cfg.LOGDIR,'output_test.pkl')
        self.standard_output_pickle = os.path.join(self.cfg.LOGDIR,'standard.pkl')

    def load_pickle(self,path):
        with open(path,'rb') as f:
            return pickle.load(f)
              
    def find_min_distance_with_standard(self,emb):
        def dist_fn(x, y):
            dist = np.sum((x-y)**2)
            return dist

        min_dists = []

        if len(emb) <= len(self.standard_emb): ## There is no need to pad the emb, it is guranteed to be played at the first frame. (if we think about the sliding window algo)
            return 0
        
        for i in range(len(emb)-len(self.standard_emb)): ## * use sliding window
            query_embs = emb[i:i+len(self.standard_emb)]   ## * compare in which window
            min_dist, _, _, _ = dtw(query_embs, self.standard_emb, dist=dist_fn) ## * the dtw yields
            min_dists.append(min_dist)
        start_frame = min_dists.index(min(min_dists)) ## * the smallest value (set it as start frame)
        return start_frame

    def __call__(self):
        for split_name,split,output_pickle in zip(['train','val','test'],[self.input_train,self.input_val,self.input_test],[self.output_train_pickle,self.output_val_pickle,self.output_test_pickle]):
            print(f"Dumping {split_name} ...")
            dataset = self.dataset[split_name]

            # this code is adjusted as "before dumping, seprate standard from test set"
            # if split_name == "test": # because we dont want to make standard's label, delete it here.
            #     del dataset["embs"][-1]
            #     del dataset["name"][-1]
            #     del dataset["video"][-1]
            #     del dataset["labels"][-1]
            assert len(dataset["name"]) == len(split),ic(len(dataset["name"]),len(split))
            for index,entry in enumerate(split):
                embs = dataset["embs"][index]
                labels = dataset["locate_module_label"][index]
                assert len(labels) == len(entry["frame_label"])
                assert len(embs) == len(entry["frame_label"])
                start_frame = self.find_min_distance_with_standard(embs)
                end_frame = start_frame + len(self.standard_emb)
                assert start_frame<= len(entry["frame_label"]) and end_frame <= len(entry["frame_label"])
                entry["locate_module_start_frame"] = start_frame
                entry["locate_module_end_frame"] = end_frame
                entry["locate_module_label"] = labels[start_frame:end_frame]

                aligned_embs = embs[start_frame:end_frame]
                assert aligned_embs.shape == self.standard_emb.shape,ic(embs.shape,self.standard_emb.shape)
                entry["embeddings"] = aligned_embs

            if self.cfg.args.overwrite:
                if split_name == "test":
                    print(f"Dumping standard ...")
                    standard_split = [split[-1]]
                    split = split[:-1]
                    assert "standard" in standard_split[0]["name"]
                    with open(self.standard_output_pickle,'wb') as f:
                        pickle.dump(standard_split,f)
                    print(f"Dumped standard to {self.standard_output_pickle}")
                    assert (len(split) + len(standard_split)) == self.original_test_length, ic(len(split),len(standard_split),self.original_test_length)
                with open(output_pickle,'wb') as f:
                    pickle.dump(split,f)
                print(f"Dumped {split_name} to {output_pickle}")
                