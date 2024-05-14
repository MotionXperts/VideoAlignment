
## SETUP
import sys
sys.path.append("/home/c1l1mo/projects/VideoAlignment")
sys.path.append("/home/c1l1mo/testings/carl")
from model.transformer.transformer import CARL as Mine_Transformer
from models.transformer import TransformerModel as CARL_Transformer
import yaml
from easydict import EasyDict as edict
import importlib.util
import sys
import torch
import random
import numpy as np
from carl_utils.parser import parse_args
import torch.nn.functional as F

def safe_div(a, b):
    out = a / b
    out[torch.isnan(out)] = 0
    return out

## Load config
with open("/home/c1l1mo/projects/VideoAlignment/result/scl_penn_action_fixed_deepclone/config.yml","r") as file:
    cfg = yaml.safe_load(file)
cfg = edict(cfg)
args = parse_args()


spec = importlib.util.spec_from_file_location("datasets", "/home/c1l1mo/testings/carl/datasets/__init__.py")
datasets = importlib.util.module_from_spec(spec)
sys.modules["datasets"] = datasets
spec.loader.exec_module(datasets)

def show_one_sample(loader):
    for i, data in enumerate(loader):
        if i == 0:
            print(data)
            break

def forward_one_sample(loader,model):
    setup_seed(7)
    for cur_iter, (originval_video,videos, _labels, seq_lens, chosen_steps, video_masks, names) in enumerate(loader):
        if cur_iter == 0:
            batch_size, num_views, num_steps, c, h, w = videos.shape
            videos = videos.view(-1, num_steps, c, h, w)
            return model(videos,None,None)

def compute_sequence_loss(embs, seq_lens, steps, masks=None):

    batch_size, num_views, num_frames, channels = embs.shape

    embs = embs.view(-1, channels) # (batch_size*num_views*num_frames, channels)
    steps = steps.view(-1)
    seq_lens = seq_lens.unsqueeze(-1).expand(batch_size, num_views, num_frames).contiguous().view(-1).float()
    input_masks = masks.view(-1, 1)*masks.view(1, -1)

    logits = torch.matmul(embs, embs.transpose(0,1)) / cfg.SCL.SOFTMAX_TEMPERATURE
    distence = torch.abs(steps.view(-1,1)/seq_lens.view(-1,1)*seq_lens.view(1,-1)-steps.view(1,-1))
    distence.masked_fill_((input_masks==0), 1e6)
    weight = torch.ones_like(logits)
    nn = torch.zeros_like(steps).long()

    # negative weight
    for b in range(batch_size):
        start = b*num_views*num_frames
        mid = start+num_frames
        end = (b+1)*num_views*num_frames
        nn[start:mid] = mid+torch.argmin(distence[start:mid,mid:end], dim=1)
        nn[mid:end] = start+torch.argmin(distence[mid:end,start:mid], dim=1)
        if "single" in cfg.SCL.NEGATIVE_TYPE:
            weight[start:end,:start].fill_(0)
            weight[start:end,end:].fill_(0)
        if "noself" in cfg.SCL.NEGATIVE_TYPE:
            weight[start:mid,start:mid] = 0
            weight[mid:end,mid:end] = 0
    weight.masked_fill_((input_masks==0), 1e-6)

    # positive weight
    label = torch.zeros_like(logits)
    if cfg.SCL.POSITIVE_TYPE == "gauss":
        pos_weight = torch.exp(-torch.square(distence)/(2*cfg.SCL.LABEL_VARIENCE)).type_as(logits)
        for b in range(batch_size):
            start = b*num_views*num_frames
            mid = start+num_frames
            end = (b+1)*num_views*num_frames
            cur_pos_weight = pos_weight[start:mid,mid:end]
            label[start:mid,mid:end] = safe_div(cur_pos_weight, cur_pos_weight.sum(dim=1, keepdim=True))
            cur_pos_weight = pos_weight[mid:end,start:mid]
            label[mid:end,start:mid] = safe_div(cur_pos_weight, cur_pos_weight.sum(dim=1, keepdim=True))

    exp_logits = torch.exp(logits)
    sum_negative = torch.sum(weight*exp_logits, dim=1, keepdim=True)

    loss = F.kl_div(torch.log(safe_div(exp_logits, sum_negative) + 1e-6), label, reduction="none")
    loss = torch.sum(loss*input_masks)
    loss = loss / torch.sum(masks)
    
    return {"loss": loss}

def compute_loss_one_sample(loader,model,cfg):
    setup_seed(7)
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(0)
        print("train loader Set epoch")
    if hasattr(train_loader.batch_sampler, 'set_epoch'):
        train_loader.batch_sampler.set_epoch(0)
        print("train batch loader Set epoch")
    for cur_iter, (originval_video,videos, _labels, seq_lens, chosen_steps, video_masks, names) in enumerate(loader):
        if cur_iter == 0:
            with torch.cuda.amp.autocast():
                num_frames = cfg.TRAIN.NUM_FRAMES
                batch_size, num_views, num_steps, c, h, w = videos.shape
                videos = videos.view(-1, num_steps, c, h, w)
                if video_masks is not None:
                    video_masks = video_masks.view(-1, 1, num_steps)
                embs = model(videos, video_masks=video_masks)
                embs = embs.view(batch_size, num_views, num_frames, embs.size(-1))
                seq_lens = seq_lens.view(batch_size, num_views)
                return compute_sequence_loss(embs,seq_lens.to(embs.device),chosen_steps.to(embs.device),video_masks.to(embs.device))

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

## Create model
def create_model(module, cfg):
    setup_seed(7)
    model = module(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], 
            output_device = args.local_rank, find_unused_parameters=True)
    return model

## Start DDP 
import carl_utils.distributed as du
torch.distributed.init_process_group(backend='nccl', init_method='env://')
du.init_distributed_training(cfg)
CARL_model = create_model(CARL_Transformer,cfg)
Mine_model = create_model(Mine_Transformer,cfg)

## load dataset
train_loader, train_emb_loader = datasets.construct_dataloader(cfg, "train")
# print("Validating Forward ...")
# carl_embs = forward_one_sample(train_loader,CARL_model)
# mine_embs = forward_one_sample(train_loader,Mine_model)

# print(torch.equal(carl_embs,mine_embs))

## Tested, make sure not to load anycheckpoint if using testings/carl (if the resnet_finetune was mismatched, it probabliy is)
# print("Validating Loss ...")
# carl_loss = compute_loss_one_sample(train_loader,CARL_model,cfg)
# mine_loss = compute_loss_one_sample(train_loader,Mine_model,cfg)
# print(carl_loss,mine_loss) 

setup_seed(7)
test = torch.rand(1,14,3,224,224)
# print(test)
# setup_seed(7)
# m = Mine_model(test)
setup_seed(7)
c = CARL_model(test)
print(c)
# print(torch.equal(m,c))



torch.distributed.destroy_process_group()