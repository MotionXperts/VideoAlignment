from dataset.skating import Skating
from dataset.pouring import Pouring
from dataset.penn_action import PennAction,ActionBatchSampler
import torch
from icecream import ic

def construct_dataloader(cfg,split,algo=None,force_test=False):
    
    if algo =="tcc":
        batch_size = cfg.TRAIN.TCC_BATCH_SIZE
    elif algo =="scl":
        batch_size = cfg.TRAIN.SCL_BATCH_SIZE
    else:
        raise Exception(f"algo {algo} not supported")

    if 'pouring'in cfg.PATH_TO_DATASET:
        dataset = Pouring(cfg,split,algo=algo)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,
                            num_workers=cfg.DATA.NUM_WORKERS,sampler=train_sampler,pin_memory=True,drop_last=True)
        eval_dataset = Pouring(cfg,split,sample_all=True)
        eval_dataloader = [torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,
                             num_workers=cfg.DATA.NUM_WORKERS,sampler=None,pin_memory=True,drop_last=True)]
    elif "penn_action" in cfg.PATH_TO_DATASET:
        dataset = PennAction(cfg,split,mode="train",algo=algo)
        if "tcc" in cfg.TRAINING_ALGO:
            sys.exit("TCC not implemented for PennAction!!")
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) 
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
                                                shuffle=True if train_sampler is None else False,
                                                num_workers=cfg.DATA.NUM_WORKERS, pin_memory=True, sampler=train_sampler,
                                                drop_last=True)
        eval_dataloader = []
        for dataset_name in cfg.DATASETS:
            eval_dataset = PennAction(cfg,split,dataset_name=dataset_name,sample_all=True,algo=algo)
            eval_dataloader.append(torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,
                                 num_workers=cfg.DATA.NUM_WORKERS,sampler=None,pin_memory=True,drop_last=True))
    else:
        dataset = Skating(cfg,split,algo=algo,force_test=force_test)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,
                            num_workers=cfg.DATA.NUM_WORKERS,sampler=train_sampler,pin_memory=True,drop_last=True)
        eval_dataset = Skating(cfg,split,sample_all=True,force_test=force_test)
        eval_dataloader = [torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,
                            num_workers=cfg.DATA.NUM_WORKERS,sampler=None,pin_memory=True,drop_last=True)]
    return dataloader,train_sampler, eval_dataloader

import unittest
from utils.config import get_cfg
import os,sys
import torch.distributed as dist

import numpy as np

class TestConstructDataloader(unittest.TestCase):
    def test_construct_dataloader(self):
        
        world_size = torch.cuda.device_count()
        self.init_process(0,world_size)

    def init_process(self,rank, world_size):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend='nccl', init_method='env://')

        cfg = get_cfg()
        cfg.TRAIN.SCL_BATCH_SIZE = 1
        cfg.TRAIN.TCC_BATCH_SIZE = 2
        cfg.DATASETS = [
            "baseball_pitch",
            "baseball_swing",
            "bench_press",
            "bowl",
            "clean_and_jerk",
            "golf_swing",
            "jumping_jacks",
            "pushup",
            "pullup",
            "situp",
            "squat",
            "tennis_forehand",
            "tennis_serve"]
        cfg.PATH_TO_DATASET = os.path.join("/home/c1l1mo/datasets", "penn_action")
        split = 'train'

        dataloader,_, eval_dataloader = construct_dataloader(cfg, split,algo="tcc")

        # Check that the dataloader and eval_dataloader are instances of DataLoader
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert isinstance(eval_dataloader, list)
        assert isinstance(eval_dataloader[0], torch.utils.data.DataLoader)
        for _ in range(5):
            item = next(iter(dataloader))
            print(item[5])

            for e in eval_dataloader:
                item = next(iter(e))
                print(item[5])
                item = next(iter(e))
                print(item[5])
            # self.assertIsInstance(item, tuple)
            # self.assertEqual(len(item), 7)


        dist.destroy_process_group()

    

if __name__ == '__main__':
    unittest.main()