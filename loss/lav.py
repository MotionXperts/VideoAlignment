import torch 
from torch import nn
from icecream import ic


class LAV():
    def __init__(self,cfg):
        self.sigma = cfg.LAV.SIGMA
        self.margin = cfg.LAV.MARGIN
        self.alpha = cfg.LAV.ALPHA
    def compute_loss(self,embs,steps,seq_lens):
        # embs: (B, T, C)
        emb_x = embs[0].unsqueeze(0)
        emb_y = embs[1].unsqueeze(0)

        steps_x = steps[0]
        steps_y = steps[1]

        seq_len_x = seq_lens[0]
        seq_len_y = seq_lens[1]

        # frame level loss
        dist_a = self.calc_distance_matrix(emb_x, emb_x).squeeze(0)
        dist_b = self.calc_distance_matrix(emb_y, emb_y).squeeze(0)

        idm_a, _ = self.inverse_idm(dist_a, steps_x, seq_len_x)
        idm_b, _ = self.inverse_idm(dist_b, steps_y, seq_len_y)

        total_loss = self.alpha * (idm_a + idm_b)

        num_frames = emb_x.size(1)
        total_loss = total_loss / num_frames

        return total_loss
    def calc_distance_matrix(self,x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        dist = torch.pow(x - y, 2).sum(3)
        return dist
    def inverse_idm(self, dist, idx, seq_len):
        grid_x, grid_y = torch.meshgrid(idx, idx)

        prob = nn.ReLU()(self.margin - dist)

        weights_orig = 1 + torch.pow(grid_x - grid_y, 2)

        diff = torch.abs(grid_x - grid_y) - (self.sigma / seq_len)
        
        _ones = torch.ones_like(diff)
        _zeros = torch.zeros_like(diff)
        weights_neg = torch.where(diff > 0, weights_orig, _zeros)

        weights_pos = torch.where(diff > 0, _zeros, _ones)
        
        idm = weights_neg * prob + weights_pos * dist

        return torch.sum(idm), idm
    
import unittest
from easydict import EasyDict as edict

class TestLAV(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'LAV': {
                'SIGMA': 10,
                'MARGIN': 1.0
            }
        }
        self.lav = LAV(edict(self.cfg))

    def test_calculate_distance_matrix(self):
        emb_a = torch.randn(10, 512)
        emb_b = torch.randn(10, 512)
        distance = self.lav.calculate_distance_matrix(emb_a, emb_b)
        self.assertEqual(distance.shape, (10, 10))

    def test_compute_loss(self):
        dist = torch.randn(10, 10)
        idx = torch.arange(10)
        seq_len = 10
        loss, idm = self.lav.compute_loss(dist, idx, seq_len)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(idm.shape, (10, 10))

if __name__ == '__main__':
    # unittest.main()
    # Configuration
    cfg = edict({
        'LAV': {
            'SIGMA': 10,
            'MARGIN': 5
        }
    })
    torch.manual_seed(0)
    # Test embeddings
    close_embeddings = torch.rand(1,5, 3)  # Embeddings close to each other
    far_embeddings = torch.rand(1,5, 3) * 10  # Embeddings far from each other

    # Initialize LAV class
    lav = LAV(cfg)

    # Calculate distance matrices
    close_dist = lav.calc_distance_matrix(close_embeddings, close_embeddings)
    far_dist = lav.calc_distance_matrix(far_embeddings, far_embeddings)

    # Compute losses
    seq_len = 5
    idx = torch.arange(seq_len)
    close_loss, _ = lav.compute_loss(close_dist, idx, seq_len)
    far_loss, _ = lav.compute_loss(far_dist, idx, seq_len)

    print(f"Loss for close embeddings: {close_loss.item()}")
    print(f"Loss for far embeddings: {far_loss.item()}")
