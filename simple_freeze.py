import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimplifiedCARL(nn.Module):
    def __init__(self, frames_per_batch=5):
        super().__init__()
        self.frames_per_batch = frames_per_batch

        # Simplified backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Simplified transformer embedding
        self.embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Simplified positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 10, 128))
        
        # Simplified transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256)
        
        # Output projection
        self.projection = nn.Linear(128, 128)

    def forward(self, x):
        B, T, C, H, W = x.shape
        num_blocks = int(math.ceil(float(T)/self.frames_per_batch))
        backbone_out = []

        for i in range(num_blocks):
            curr_idx = i * self.frames_per_batch
            cur_steps = min(T-curr_idx, self.frames_per_batch)
            curr_data = x[:, curr_idx:curr_idx+cur_steps]
            curr_data = curr_data.contiguous().view(-1, C, H, W)
            
            with torch.no_grad():
                self.backbone.eval()
                curr_emb = self.backbone(curr_data)
            
            curr_emb = curr_emb.detach().requires_grad_()
            _, out_c, out_h, out_w = curr_emb.size()
            curr_emb = curr_emb.contiguous().view(B, cur_steps, out_c * out_h * out_w)
            backbone_out.append(curr_emb)
        backbone_output = torch.cat(backbone_out, dim=1).detach().requires_grad_()
        # x = torch.cat(backbone_output, dim=1)

        # Reshape and pass through backbone
        # x = x.view(B*T, C, H, W)
        # with torch.no_grad():
        #     self.backbone.eval()
        #     x = self.backbone(x)
        # curr_emb = x.detach().requires_grad_()
        # x = curr_emb.view(B, T, -1)
        
        # Embedding
        x = self.embed(backbone_output)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer layer
        x = self.transformer_layer(x)
        
        # Projection and normalization
        x = self.projection(x)
        x = F.normalize(x, dim=-1)
        
        return x, curr_emb  # Return both output and backbone_out

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
    if input_tensor.grad is not None:
        print(f"Input gradient norm: {input_tensor.grad.norm()}")
    else:
        print("No gradient computed for input tensor")

    # Check backbone output gradient
    if backbone_out.grad is not None:
        print(f"Backbone output gradient norm: {backbone_out.grad.norm()}")
    else:
        print("No gradient computed for backbone output")

# Test with simplified model
simplified_model = SimplifiedCARL(frames_per_batch=5)
input_tensor = torch.randn(1, 10, 3, 224, 224, requires_grad=True)
print("Testing with simplified CARL model:")
check_gradient_flow(simplified_model, input_tensor)

# Your existing code for testing the actual CARL model...