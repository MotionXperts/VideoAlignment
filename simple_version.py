import torch
import torch.nn as nn
from utils.parser import parse_args,load_config
import os
import torch.distributed as dist

class SimplifiedCARL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = x.reshape(-1, 3, 224, 224)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def hook_fn(module, grad_input, grad_output):
    print(f"Gradient norm for {module.__class__.__name__}: {grad_output[0].norm()}")

def check_gradient_flow(model, input_tensor):
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_backward_hook(hook_fn))

    # Forward pass
    output, backbone_out = model(input_tensor)
    
    # Compute loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Check input gradient
    if backbone_out.grad is not None:
        print(f"Input gradient norm: {backbone_out.grad.norm()}")
    else:
        print("No gradient computed for input tensor")

# Test with simplified model
# simplified_model = SimplifiedCARL()
# input_tensor = torch.randn(1, 10,3, 224, 224, requires_grad=True)
# print("Testing with simplified model:")
# check_gradient_flow(simplified_model, input_tensor)

# Test with your actual CARL model
from model.carl_transformer.transformer import TransformerModel
args = parse_args()
cfg = load_config(args)
cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
cfg.args = args


# Initialize the model
dist.init_process_group(backend='nccl', init_method='env://')
model = TransformerModel(cfg)
input_tensor = torch.randn(1, 10,3, 224, 224, requires_grad=True)
torch.cuda.set_device(args.local_rank)
model = model.cuda()
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

carl_model = TransformerModel(cfg)
print("\nTesting with actual CARL model:")
check_gradient_flow(carl_model, input_tensor)

dist.destroy_process_group()