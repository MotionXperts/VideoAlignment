# coding=utf-8
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

import wandb
import os,sys
import logging 
logger = logging.getLogger(__name__)

def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    return e / np.sum(e)

class KendallsTau(object):
    """Calculate Kendall's Tau."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.downstream_task = True
        self.stride = cfg.EVAL.KENDALLS_TAU_STRIDE
        self.dist_type = cfg.EVAL.KENDALLS_TAU_DISTANCE
        if cfg.MODEL.L2_NORMALIZE:
            self.temperature = 0.1
        else:
            self.temperature = 1

        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',filename=os.path.join(cfg.LOGDIR,'stdout.log'))

    def evaluate(self, dataset, cur_epoch, summary_writer,split = 'val'):
        """Labeled evaluation."""
        # train_embs = dataset['train_dataset']['embs']

        # self.get_kendalls_tau(
        #         train_embs,
        #         cur_epoch, summary_writer,
        #         '%s_train' % dataset['name'], visualize=True)

        val_embs = dataset['embs']
        val_labels = dataset['labels']
        val_names = dataset['name']

        tau = self.get_kendalls_tau(val_embs,val_labels,val_names, cur_epoch, summary_writer, split, visualize=False)
        return tau

    def get_kendalls_tau(self, embs_list,labels_list,val_names, cur_epoch, summary_writer, split, visualize=True):

        query = np.random.randint(0,len(embs_list))
        candidate = np.random.randint(0,len(embs_list))
        while query == candidate:
            candidate = np.random.randint(0,len(embs_list))

        """Get nearest neighbours in embedding space and calculate Kendall's Tau."""
        num_seqs = len(embs_list)
        taus = np.zeros((num_seqs * (num_seqs - 1)))
        idx = 0
        if split == "train":
            self.compute_labels = 0
        else:
            try:
                self.compute_labels = self.cfg.EVAL.KENDALLS_TAU_COMPUTE_LABELS
            except:
                self.compute_labels = 0

        for i in range(num_seqs):
            query_feats = embs_list[i][::self.stride]
            valid_frames = np.where(labels_list[i]>=self.compute_labels)[0]
            query_feats = embs_list[i][valid_frames]
                
            for j in range(num_seqs):
                if i == j: continue
                candidate_feats = embs_list[j][::self.stride]
                candi_valid_frames = np.where(labels_list[j]>=self.compute_labels)[0]
                candidate_feats = embs_list[j][candi_valid_frames]
                dists = cdist(query_feats, candidate_feats, self.dist_type)
                nns = np.argmin(dists, axis=1)
                # if i == 0 and j ==4 :
                #     logger.info(f"comparing {val_names[i]} and {val_names[j]}")
                #     logger.info(f"query feats: {query_feats}")
                #     logger.info(f"candidate feats: {candidate_feats}")
                #     logger.info(f"nns: {nns}")
                if visualize:
                    if (i==0 and j == 1) or (i < j and num_seqs == 14):
                        sim_matrix = []
                        for k in range(len(query_feats)):
                            sim_matrix.append(softmax(-dists[k], t=self.temperature))
                        sim_matrix = np.array(sim_matrix, dtype=np.float32)
                        if summary_writer is not None:
                            summary_writer.add_image(f'{split}/sim_matrix_{i}_{j}', sim_matrix.T, cur_epoch, dataformats='HW')
                taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
                # logger.info(f"Kendall's Tau ({self.compute_labels}): %.4f" % taus[idx])
                idx += 1
        # Remove NaNs.
        taus = taus[~np.isnan(taus)]
        tau = np.mean(taus)

        logger.info(f"[{split}] Kendall's Tau ({self.compute_labels}): %.4f" % tau)

        try:
            summary_writer.add_scalar('kendalls_tau/%s_align_tau' % split, tau, cur_epoch)
            wandb.log({f"kendalls_tau/{split}_align_tau": tau , "custom_step": cur_epoch})
        except:
            pass
        return tau

import unittest
from easydict import EasyDict as edict

class TestKendallsTau(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'EVAL': {
                'KENDALLS_TAU_STRIDE': 2,
                'KENDALLS_TAU_DISTANCE': 'euclidean',
                'KENDALLS_TAU_COMPUTE_LABELS': 1    
            },
            'MODEL': {
                'L2_NORMALIZE': True
            }
        }
        self.cfg = edict(self.cfg)
        self.kt = KendallsTau(self.cfg)

    def test_get_kendalls_tau(self):
        torch.manual_seed(0)
        np.random.seed(0)
        embs_list = [
            torch.randn(10, 512),
            torch.randn(10, 512),
        ]
        labels_list = [
            np.array([0,0,0,1,1,1,1,0,0,0]),
            np.array([0,0,0,0,1,1,1,1,1,1,0]),
        ]
        cur_epoch = 1
        summary_writer = None
        split = 'val'
        print(embs_list,labels_list)
        tau = self.kt.get_kendalls_tau(embs_list, labels_list, cur_epoch, summary_writer, split)
        self.assertIsInstance(tau, float)

if __name__ == '__main__':
    unittest.main()