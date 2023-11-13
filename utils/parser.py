import os 
import yaml 
import argparse
from utils.config import get_cfg
import sys
from easydict import EasyDict 

"""
Parse the input flags. 
Most importantly, read the config file path. Consturct one folder to store it if omit or missing, otherwise use it.
"""

USER = os.environ['USER']

def parse_args():
    parser = argparse.ArgumentParser(description="TCC training pipeline.")
    parser.add_argument('--local_rank', default=0, type=int, help='rank in local processes')

    parser.add_argument('--workdir', type=str, default=f'/home/{USER}/datasets', help='Path to datasets and pretrained models.')
    parser.add_argument('--logdir', type=str, default=None, help='Path to logs.')
    parser.add_argument('--visualize', action='store_true',
                        default=False, help='Visualize images, gradients etc. \
                        Switched off by for default to speed training up and \
                        takes less memory.')
    parser.add_argument(
        "--cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--demo_or_inference",
        help="Whether to run demo or inference",
        default="demo",
        type=str,
    )

    parser.add_argument(
        "--K",
        help="Number of videos to generate",
        default=1,
        type=int,
    )

    parser.add_argument('--record', action='store_true',
                        default=False)

    parser.add_argument('--debug',action="store_true",default=False)
    
    parser.add_argument('--ckpt',
                        default=None,
                        type=str,
                        help='Path to checkpoint')
    parser.add_argument("--generate",
                        action="store_true",
                        default=False)
    parser.add_argument("--carl",action="store_true",default=False)

    return parser.parse_args()

def to_dict(config):
    if isinstance(config, list):
        return [to_dict(c) for c in config]
    elif isinstance(config, EasyDict):
        return dict([(k, to_dict(v)) for k, v in config.items()])
    else:
        return config

def load_config(args):
    cfg=get_cfg()
    print(f'CONFIG FILE: {args.cfg_file} EXISTS? {os.path.exists(args.cfg_file)}')
    if args.cfg_file is not None and os.path.exists(args.cfg_file):
        print(f'Using config from {(args.cfg_file)}.')
        with open(args.cfg_file, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        cfg.update(config_dict)
        config_file = args.cfg_file
    elif args.cfg_file is not None:
        print(f"{args.cfg_file} not found")
        sys.exit(1)

    else:
        cfg.LOGDIR = os.path.join(cfg.LOGDIR,cfg.PATH_TO_DATASET)
        os.makedirs(cfg.LOGDIR, exist_ok=True)
        config_file = os.path.join(cfg.LOGDIR,'config.yaml')
        ## dump config to yaml file
        if os.path.exists(config_file):
            print("config.yaml already exists")
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            cfg.update(config_dict)
    # update config file
    cfg.VISUALIZATION_DIR = os.path.join(cfg.LOGDIR,"visualization")
    # with open(config_file, 'w') as f:
    #     config = dict([(k, to_dict(v)) for k, v in cfg.items()])
    #     yaml.safe_dump(config, f,default_flow_style=False)
    os.makedirs(cfg.VISUALIZATION_DIR,exist_ok=True)

    return cfg

if __name__ == "__main__":
    print("testing parser functionallity")
    args = parse_args()
    print(f'args: ' , args)