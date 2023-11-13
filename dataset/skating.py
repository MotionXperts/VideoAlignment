# coding=utf-8
import os
import math 
import pickle
import torch
import torch.nn.functional as F
from torchvision.io import read_video
import numpy as np
# import utils.dist as du
import json
from icecream import ic
import time

# import utils.logging as logging
from dataset.data_augment import create_data_augment,create_ssl_data_augment,create_simple_augment

# logger = logging.get_logger(__name__)

class Skating(torch.utils.data.Dataset):
    def __init__(self,cfg,split,sample_all=False,algo=None,train=None):
        # cfg.PATH_TO_DATASET = '/home/c1l1mo/datasets/new_boxing_no_overlapped'

        self.cfg = cfg
        self.split = split

        self.sample_all = sample_all
        self.num_contexts = cfg.DATA.NUM_CONTEXTS

        self.mode = split
        self.algo = algo

        self.train = train

        with open(os.path.join(cfg.PATH_TO_DATASET, self.mode + '.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)
        
        if not self.sample_all:
            # logger.info(f"{len(self.dataset)} {self.split} samples of Pouring dataset have been read.")
            seq_lens = [data['seq_len'] for data in self.dataset]
            hist, bins = np.histogram(seq_lens, bins='auto')
            # if du.is_root_proc():
            print(list(bins.astype(np.int)))
            print(list(hist))

        if self.mode=="train" and cfg.TRAINING_ALGO == 'classification':
            num_train = max(1, int(cfg.DATA.FRACTION * len(self.dataset)))
            self.dataset = self.dataset[:num_train]

        self.num_frames = cfg.TRAIN.NUM_FRAMES
        # Perform data-augmentation
        if self.cfg.SSL and self.mode=="train" :
            self.data_preprocess,self.b4_norm = create_ssl_data_augment(cfg, augment=True)
        elif self.mode=="train" and not self.cfg.DATA.SIMPLE_PREPROCESS:
            self.data_preprocess = create_data_augment(cfg, augment=True)
        elif hasattr(self.cfg.DATA, "SIMPLE_PREPROCESS") and self.cfg.DATA.SIMPLE_PREPROCESS:
            self.data_preprocess = lambda x:((x * 255.0)  / 127.5) - 1.0
        else:
            self.data_preprocess = create_data_augment(cfg, augment=False)

        if 'tcn' in cfg.TRAINING_ALGO:
            self.num_frames = self.num_frames // 2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        id = self.dataset[index]["id"]
        
        name = self.dataset[index]["name"]
        frame_label = self.dataset[index]["frame_label"]
        seq_len = self.dataset[index]["seq_len"]
        if self.train is not None:
            video = torch.from_numpy(self.train[index]["video"])
            print(self.train[index]['name'])
        else:
            video_file = os.path.join(self.cfg.PATH_TO_DATASET, self.dataset[index]["video_file"])
            video, _, info = read_video(video_file, pts_unit='sec')
        video = video.permute(0,3,1,2).float() / 255.0 # T H W C -> T C H W, [0,1] tensor

        skeleton = 0
        if hasattr(self.cfg.DATA, "SKELETON") and self.cfg.DATA.SKELETON:
            if "skeleton_file" in self.dataset[index]:
                skeleton = {}
                skeleton_file = self.dataset[index]["skeleton_file"] 
                with open(skeleton_file,"r") as file:
                    json_skeletons = json.load(file)
                for json_skeleton in json_skeletons:
                    image_id = json_skeleton["image_id"]
                    skeleton[image_id] = json_skeleton["keypoints"]
                previous_image_id = -1

                tmp_skeleton = skeleton.copy()

                # if "0000.jpg" not in skeleton:
                #     tmp_skeleton["0000.jpg"] = skeleton["0001.jpg"]
                #     previous_image_id = 0 

                for image_id in (skeleton):
                    int_image_id = int(image_id.split(".jpg")[0])
                    if previous_image_id +1 !=int_image_id:
                        while previous_image_id+1 != int_image_id:
                            ## this will mend missing frame using next known frame (e.g: if 22 and 25 are known, then 23 and 24 will be filled with 25)
                            tmp_skeleton[str(previous_image_id+1).zfill(4)+".jpg"] = skeleton[image_id] 
                            previous_image_id +=1
                    previous_image_id = int_image_id

                tmp_skeleton_length = len(tmp_skeleton)
                while tmp_skeleton_length < seq_len:
                    tmp_skeleton[str(tmp_skeleton_length).zfill(4)+".jpg"] = skeleton[image_id]
                    tmp_skeleton_length +=1


                tmp_skeleton = dict(sorted(tmp_skeleton.items(), key=lambda item: int(item[0].split(".jpg")[0])))

                skeleton = torch.from_numpy(np.array(list(tmp_skeleton.values()))).type_as(video)

        if self.train is None:
            assert len(video) == seq_len
            assert len(video) == len(frame_label)

        if self.cfg.SSL and not self.sample_all and self.algo=="scl":
            names = [name, name]
            steps_0, chosen_step_0, video_mask0 = self.sample_frames(seq_len, self.num_frames)
            view_0 = self.data_preprocess(video[steps_0.long()])
            label_0 = frame_label[chosen_step_0.long()]
            steps_1, chosen_step_1, video_mask1 = self.sample_frames(seq_len, self.num_frames, pre_steps=steps_0)
            view_1 = self.data_preprocess(video[steps_1.long()])
            label_1 = frame_label[chosen_step_1.long()]
            videos = torch.stack([view_0, view_1], dim=0)
            labels = torch.stack([label_0, label_1], dim=0)
            seq_lens = torch.tensor([seq_len, seq_len])
            chosen_steps = torch.stack([chosen_step_0, chosen_step_1], dim=0)
            video_mask = torch.stack([video_mask0, video_mask1], dim=0)
            skeleton = 0
            return videos, videos, labels, seq_lens, chosen_steps, video_mask, names,skeleton

        elif not self.sample_all:
            steps, chosen_steps, video_mask = self.sample_frames(seq_len, self.num_frames)
        else:
            steps = torch.arange(0, seq_len, self.cfg.DATA.SAMPLE_ALL_STRIDE)
            seq_len = len(steps)
            chosen_steps = steps.clone()
            video_mask = torch.ones(seq_len)
        
        video = video[steps.long()]
        ## not flipping skeleton right now.
        if hasattr(self.cfg.DATA, "SKELETON") and self.cfg.DATA.SKELETON:
            skeleton = skeleton[steps.long()]
        ## not flipping skeleton right now.
        original_video = video.clone()
        video = self.data_preprocess(video)
        video = video.unsqueeze(0)
        video_mask=video_mask.unsqueeze(0)
        label = frame_label[chosen_steps.long()]

        return original_video,video, label, torch.tensor(seq_len), chosen_steps, video_mask, name,skeleton
        # return video, label, torch.tensor(seq_len), steps, video_mask, name ## if we return steps (40) instead of chosen steps (20), there will be expandsion prob in tcc loss.

    def sample_frames(self, seq_len, num_frames, pre_steps=None):
        # When dealing with very long videos we can choose to sub-sample to fit
        # data in memory. But be aware this also evaluates over a subset of frames.
        # Subsampling the validation set videos when reporting performance is not
        # recommended.
        sampling_strategy = self.cfg.DATA.SAMPLING_STRATEGY
        pre_offset = min(pre_steps) if pre_steps is not None else None
        
        if sampling_strategy == 'offset_uniform':
            # Sample a random offset less than a provided max offset. Among all frames
            # higher than the chosen offset, randomly sample num_frames
            if seq_len >= num_frames:
                steps = torch.randperm(seq_len) # Returns a random permutation of integers from 0 to n - 1.
                steps = torch.sort(steps[:num_frames])[0]
                steps = torch.tensor([0,1,3,5,6,7,8,9,13,14,15,16,17,19,20,21,22,23,25,26])
            else:
                steps = torch.arange(0, num_frames)
        elif sampling_strategy == 'time_augment':
            num_valid = min(seq_len, num_frames)
            expand_ratio = np.random.uniform(low=1.0, high=self.cfg.DATA.SAMPLING_REGION) if self.cfg.DATA.SAMPLING_REGION>1 else 1.0

            block_size = math.ceil(expand_ratio*num_valid)
            if pre_steps is not None and self.cfg.DATA.CONSISTENT_OFFSET != 0:
                shift = int((1-self.cfg.DATA.CONSISTENT_OFFSET)*num_valid)
                offset = np.random.randint(low=max(0, min(seq_len-block_size, pre_offset-shift)), high=max(1, min(seq_len-block_size+1, pre_offset+shift+1)))
            else:
                offset = np.random.randint(low=0, high=max(seq_len-block_size, 1))
            steps = offset + torch.randperm(block_size)[:num_valid]
            steps = torch.sort(steps)[0]
            if num_valid < num_frames:
                steps = F.pad(steps, (0, num_frames-num_valid), "constant", seq_len)
        else:
            raise ValueError('Sampling strategy %s is unknown. Supported values are '
                            'stride, offset_uniform .' % sampling_strategy)

        video_mask = torch.ones(num_frames)
        video_mask[steps<0] = 0
        video_mask[steps>=seq_len] = 0
        # Store chosen indices.
        chosen_steps = torch.clamp(steps.clone(), 0, seq_len - 1)
        if self.num_contexts == 1:
            steps = chosen_steps
        else:
            # Get multiple context steps depending on config at selected steps.
            context_stride = self.cfg.DATA.CONTEXT_STRIDE
            steps = steps.view(-1,1) + context_stride*torch.arange(-(self.num_contexts-1), 1).view(1,-1)
            steps = torch.clamp(steps.view(-1), 0, seq_len - 1)

        return steps, chosen_steps, video_mask

import unittest
from easydict import EasyDict as edict
import random
import yaml

class TestSkating(unittest.TestCase):
    

    def setup_seed(self,seed):
        random.seed(seed)                          
        np.random.seed(seed)                       
        torch.manual_seed(seed)                    
        torch.cuda.manual_seed(seed)               
        torch.cuda.manual_seed_all(seed)           
        torch.backends.cudnn.deterministic = True  

    def setUp(self):
        with open("/home/c1l1mo/projects/tcc/result/Skating/axel_trimmed/new_pickles/train.pkl","rb") as file:
            train = pickle.load(file)

        self.setup_seed(7)
        self.cfg = edict({
            'PATH_TO_DATASET': '/home/c1l1mo/datasets/processed_axel_trimmed',
            'DATA': {
                'NUM_CONTEXTS': 1,
                'SAMPLING_STRATEGY': 'offset_uniform',
                'SAMPLE_ALL_STRIDE': 2,
                'FRACTION': 1.0,
                'SKELETON': True,
                'SIMPLE_PREPROCESS': True,
            },
            'TRAIN': {
                'NUM_FRAMES': 20
            },
            'SSL': False,
            'TRAINING_ALGO': 'classification',
            'AUGMENTATION':{
                'BRIGHTNESS': True,
                'BRIGHTNESS_MAX_DELTA': 0.8,
                'CONTRAST': True,
                'CONTRAST_MAX_DELTA': 0.8,
                'HUE': True,
                'HUE_MAX_DELTA': 0.2,
                'RANDOM_CROP': True,
                'RANDOM_FLIP': True,
                'SATURATION': True,
                'SATURATION_MAX_DELTA': 0.8,
                'STRENGTH': 1.0,
            },
            'IMAGE_SIZE': 224
        })
        
        with open("/home/c1l1mo/projects/VideoAlignment/result/conv_processed_axel_trimmed2/config.yaml",'r') as f:
            self.cfg = edict(yaml.safe_load(f))
            self.cfg.PATH_TO_DATASET = '/home/c1l1mo/datasets/processed_axel_trimmed'
        self.split = 'train'
        self.sample_all = False
        self.dataset = Skating(self.cfg, self.split, self.sample_all, train=train)

    def test_len(self):
        self.assertIsInstance(len(self.dataset), int)

    def test_getitem(self):
        np.set_printoptions(threshold=np.inf)
        index = 12
        print("fetching ...")
        origin_video,video, label, seq_len, chosen_steps, video_mask, name,skeleton = self.dataset[index]
        self.assertIsInstance(video, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertIsInstance(seq_len, torch.Tensor)
        self.assertIsInstance(chosen_steps, torch.Tensor)
        self.assertIsInstance(video_mask, torch.Tensor)
        self.assertIsInstance(name, str)
        ic(name,video[0].permute(0,2,3,1),video[0].shape)
        ic(chosen_steps)

if __name__ == '__main__':
    unittest.main()