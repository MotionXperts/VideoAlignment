import os,sys
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import torch
import torch.distributed as dist
from utils.parser import parse_args,load_config
from utils.nancy_result import *

from model import build_model
from model.carl_transformer.transformer import TransformerModel
import torch

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import inspect

def enable_gradient_flow(module):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            return output.requires_grad_(True)
        elif isinstance(output, tuple):
            return tuple(o.requires_grad_(True) if isinstance(o, torch.Tensor) else o for o in output)
        else:
            return output
    
    for name, child in module.named_children():
        if not list(child.children()):  # if it's a leaf module
            child.register_forward_hook(hook)
        else:
            enable_gradient_flow(child)

def check_inplace_operations(module, prefix=''):
    inplace_ops = []
    
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        
        # Check for in-place ReLU
        if isinstance(child, nn.ReLU) and child.inplace:
            inplace_ops.append(f"{child_prefix}: in-place ReLU")
        
        # Recursively check child modules
        inplace_ops.extend(check_inplace_operations(child, child_prefix))
    
    # Check for other potential in-place operations in the forward method
    if hasattr(module, 'forward') and callable(module.forward):
        forward_src = inspect.getsource(module.forward)
        if '._' in forward_src or 'inplace=True' in forward_src:
            inplace_ops.append(f"{prefix}: Potential in-place operation in forward method")
    
    return inplace_ops

def convert_relu_to_non_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            convert_relu_to_non_inplace(child)


def compute_saliency_maps(model, input_video, target_idx):
    model.train()

    enable_gradient_flow(model)  # Enable gradient flow for all modules
    input_video.requires_grad_()

    for param in model.parameters():
        param.requires_grad = True
    
    # Enable anomaly detection
    with torch.autograd.detect_anomaly():
        # Forward pass with hooks
        hooks = []
        intermediate_outputs = []
        
        def hook_fn(module, input, output):
            intermediate_outputs.append(output)
        
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn))
        
        output = model(input_video)
        
        for hook in hooks:
            hook.remove()
        
        # Select the target
        target = output[0, 0, target_idx]
        
        print(f"Input shape: {input_video.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Target value: {target.item()}")
        
        # Check intermediate outputs
        # for i, intermediate_output in enumerate(intermediate_outputs):
        #     print("type of intermediate_output",type(intermediate_output))
        #     print(f"Intermediate output {i} shape: {intermediate_output.shape}")
        #     print(f"Intermediate output {i} requires_grad: {intermediate_output.requires_grad}")
        #     print(f"Intermediate output {i} is_leaf: {intermediate_output.is_leaf}")
        
        # Compute gradients
        try:
            target.backward()
        except Exception as e:
            print(f"Error in backward pass: {e}")

        if input_video.grad is not None:
            print(f"Input gradient norm: {input_tensor.grad.norm()}")
        else:
            print("No gradient computed for input tensor")
            return
        
        # Check gradients
        if input_video.grad is None:
            print("Gradients were not computed. Debugging:")
            print(f"requires_grad: {input_video.requires_grad}")
            print(f"is_leaf: {input_video.is_leaf}")
            
            # Check gradients of model parameters and intermediate outputs
            print("Checking gradients of model parameters and intermediate outputs:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad shape {param.grad.shape}, grad sum: {param.grad.sum()}")
                else:
                    print(f"{name}: No grad")
            
            for i, intermediate_output in enumerate(intermediate_outputs):
                if intermediate_output.grad is not None:
                    print(f"Intermediate output {i}: grad shape {intermediate_output.grad.shape}, grad sum: {intermediate_output.grad.sum()}")
                else:
                    print(f"Intermediate output {i}: No grad")
            
            raise ValueError("Gradients could not be computed.")
    
    # Generate saliency map
    saliency, _ = torch.max(input_video.grad.data.abs(), dim=1)
    return saliency

def visualize_saliency(video, saliency_maps):
    # Assuming video shape is (1, T, C, H, W) and saliency_maps shape is (1, T, H, W)
    T = video.shape[1]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for t in range(min(T, 5)):  # Visualize up to 5 frames
        # Original frame
        axes[0, t].imshow(video[0, t].permute(1, 2, 0).cpu().numpy())
        axes[0, t].axis('off')
        axes[0, t].set_title(f'Frame {t}')
        
        # Saliency map
        axes[1, t].imshow(saliency_maps[0, t].cpu().numpy(), cmap='hot')
        axes[1, t].axis('off')
        axes[1, t].set_title(f'Saliency Map {t}')
    
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.args = args
    
    # Initialize the model
    dist.init_process_group(backend='nccl', init_method='env://')
    model = TransformerModel(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

    convert_relu_to_non_inplace(model)
    inplace_operations = check_inplace_operations(model)


    if inplace_operations:
        print("In-place operations found:")
        for op in inplace_operations:
            print(op)
    else:
        print("No in-place operations found in the model.")
    
    # Generate a sample input video
    # Assuming 1 video in the batch, 10 frames, 3 channels, 224x224 resolution
    input_video = torch.randn(1, 10, 3, 224, 224)
    
    try:
        saliency_maps = compute_saliency_maps(model, input_video, target_idx=0)
        visualize_saliency(input_video, saliency_maps)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check the model architecture and ensure gradients can be computed.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()