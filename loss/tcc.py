import torch
import sys,os
import logging
import wandb
import numpy as np
from icecream import ic
from torchvision.utils import make_grid
from torchvision import transforms as t
import utils.dist as du

logger = logging.getLogger(__name__)

class TCC():
    def __init__(self,cfg):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',filename=os.path.join(cfg.LOGDIR,'stdout.log'))
        self.label_smoothing = cfg.TCC.LABEL_SMOOTHING
        self.temperature = cfg.TCC.SOFTMAX_TEMPERATURE
        self.lambda_ = cfg.TCC.VARIANCE_LAMBDA
        self.normalize_indices = cfg.TCC.NORMALIZE_INDICES
    def compute_cycle_loss(self,emb_x,emb_y,DEBUG=False,images=None,summary_writer=None,epoch=None,split="train"):
        """
        1. Find the distance for emb_x in emb_y (using l2)
        2. softmax the distance then multiply by emb_y
        3. use 2. to find the distance for emb_x as logits
        4. construct smoothed labels 
        """
        num_steps, D = emb_x.shape

        distance = -1 * torch.cdist(emb_x,emb_y,p=2).pow(2)
        distance = distance / D / self.temperature


        sftmax = torch.softmax(distance,dim=-1)
        emb_y = torch.matmul(sftmax,emb_y)
        logits = -1 * torch.cdist(emb_y,emb_x,p=2).pow(2) / D / self.temperature

        if DEBUG and du.is_root_proc() and summary_writer is not None:
            ## images: B , T , H , W
            queries = images[0]
            candidates = images[1]
            nn = torch.argmax(sftmax,dim=-1)

            ## retrieve the nearest neighbor for each query image
            retrieved = candidates[nn]
            ## find the nearest neight back to the query image
            reverse_nn = torch.argmax(torch.softmax(logits,dim=-1),dim=-1)

            new_order = nn
            find_back_order = [reverse_nn[i].item() for i in new_order]
            find_back  = queries[find_back_order]

            resize = t.Resize((224,224))
            queries = resize(queries)
            retrieved = resize(retrieved)
            find_back = resize(find_back)
            candidates = resize(candidates)

            render = make_grid(torch.cat([queries,retrieved,find_back,candidates],dim=0),nrow=num_steps)


            ## create heapmap for the softmax
            sftmax = (sftmax).detach().cpu().numpy().astype(np.float32)
            logits_map = (torch.softmax(logits,dim=-1)).detach().cpu().numpy().astype(np.float32)

            sftmax = sftmax
            logits_map = logits_map

            summary_writer.add_image(f"{split}/name",render,global_step=epoch)
            summary_writer.add_image(f"{split}/nnForQ",sftmax,dataformats="HW", global_step=epoch)
            summary_writer.add_image(f"{split}/nnforV",logits_map,dataformats="HW",global_step=epoch)

            dummy = [[0,1,0],[1,0,0],[0,0,1]]
            dummy = np.array(dummy).astype(np.float32)
            summary_writer.add_image(f"dummy",dummy,dataformats="HW",global_step=epoch)
            
            
            # images = wandb.Image(render, caption="None")
            # wandb.log({"images": images})

        labels = torch.diag(torch.ones(num_steps)).type_as(logits)
        ## label smoothing
        if self.label_smoothing:
            labels = (1-num_steps*self.label_smoothing/(num_steps-1))*labels + \
                            self.label_smoothing/(num_steps-1)*torch.ones_like(labels)
        return logits,labels
    
    def regression_loss(self,logits,labels,steps,seq_lens,DEBUG=False):
        """
        1. Normalize the steps by seq_lens (to mitigate the influence of long seq_lens)
        2. softmax logits to obtain beta
        3. calculate i by multiplying labels and steps
        4. calculate mean by multiplying beta and steps
        5. calculate variance by multiplying beta and (i - mean)^2
        6. calculate Lcbr
        """
        if self.normalize_indices:
            steps = steps / seq_lens.unsqueeze(1)
        beta = torch.softmax(logits,dim=-1)
        i = torch.sum(steps*labels,dim=-1)
        mean = torch.sum(steps*beta,dim=-1)
        variance = torch.sum(torch.square(steps-mean.unsqueeze(1)) * beta ,dim=-1)
        # if DEBUG:
        #     logger.info(f"beta: \n{beta}")
        #     logger.info(f"logits: \n{logits}")
        #     logger.info(f"{torch.square(steps - mean.unsqueeze(-1)) * beta}\nsteps: \n{steps}\nmean: {mean}")
        log_variance = torch.log(variance)
        Lcbr = torch.mean(torch.square(i-mean) / variance + self.lambda_ * log_variance)
        return Lcbr
        
    def compute_loss(self,embs,seq_lens,steps,mask=None,batch_size=2,DEBUG=False,images=None,summary_writer=None,epoch=None,split="train"):
        """
        TCC computes loss based on the following equation:
            1. find the soft nearest neighbor for u in v, this is acheived by softmaxing
            2. compute betak, the formula is given as exp(-d(v,uk)) / sum(exp(-d(v,uj)))
        """
        steps = steps.to(embs.device)
        seq_lens = seq_lens.to(embs.device)

        logits_list = []
        labels_list = []
        steps_list = []
        seq_lens_list = []

        B , T , D = embs.shape
        for i in range((batch_size)):
            for j in range((batch_size)):
                if i ==j:
                    continue
                if i==0 and j==1:
                    DEBUG = True
                else:
                    DEBUG = False
                logits,labels = self.compute_cycle_loss(embs[i],embs[j],DEBUG=DEBUG,images=images,summary_writer=summary_writer,epoch=epoch,split=split)
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(steps[i].unsqueeze(0).expand(T,T))
                seq_lens_list.append(seq_lens[i].view(1,).expand(T))
        logits_list = torch.cat(logits_list,dim=0)
        labels_list = torch.cat(labels_list,dim=0)
        steps_list = torch.cat(steps_list,dim=0)
        seq_lens_list = torch.cat(seq_lens_list,dim=0)

        # if DEBUG:
        #     logger.info(f"logits_list: {logits_list}, labels_list: {labels_list}, steps_list: {steps_list}, seq_lens_list: {seq_lens_list}")

        loss = self.regression_loss(logits_list,labels_list,steps_list,seq_lens_list,DEBUG)
        return loss

import unittest

class TestTCC(unittest.TestCase):
    def test_compute_loss(self):
        # Create a sample input tensor
        B, T, D = 2, 50, 128
        embs = torch.randn((B, T, D))
        steps = torch.randint(low=0, high=T, size=(B, T))
        seq_lens = torch.randint(low=0, high=T, size=(B,))

        # Create a TCC object with default configuration
        cfg = type('', (), {})()
        cfg.TCC = type('', (), {})()
        cfg.TCC.LABEL_SMOOTHING = 0.1
        cfg.TCC.SOFTMAX_TEMPERATURE = 1.0
        cfg.TCC.VARIANCE_LAMBDA = 0.1
        tcc = TCC(cfg)

        # Compute the loss for the sample input tensor
        loss = tcc.compute_loss(embs, steps, seq_lens)

        # Check that the loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())

if __name__ == '__main__':
    unittest.main()
