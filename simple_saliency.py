import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

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
        frame_indices = []

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
            frame_indices.extend(range(curr_idx, curr_idx + cur_steps))

        backbone_output = torch.cat(backbone_out, dim=1)
        
        # Embedding
        x = self.embed(backbone_output)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer layer
        x = self.transformer_layer(x)
        
        # Projection and normalization
        x = self.projection(x)
        x = F.normalize(x, dim=-1)
        
        return x, backbone_output, frame_indices

def compute_saliency_map(model, input_tensor, target_frame_idx, target_batch_idx=0):
    model.eval()
    input_tensor.requires_grad_()
    
    output, backbone_output, frame_indices = model(input_tensor)
    
    # Find which position in backbone_output corresponds to our target frame
    target_pos = frame_indices.index(target_frame_idx)
    
    # Select the output for the target frame
    target_output = output[target_batch_idx, target_pos]
    
    # Compute loss (sum of output elements)
    loss = target_output.sum()
    
    # Backward pass
    loss.backward()
    
    # Get the gradients for the input tensor
    gradients = target_output.grad[target_batch_idx, target_frame_idx]
    
    # Compute saliency map
    saliency_map = gradients.abs().sum(dim=0)
    
    return saliency_map

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and input tensor
    model = SimplifiedCARL(frames_per_batch=5)
    input_tensor = torch.randn(1, 10, 3, 224, 224, requires_grad=True)
    
    # Choose a target frame
    target_frame_idx = 5
    
    # Compute saliency map
    saliency_map = compute_saliency_map(model, input_tensor, target_frame_idx)
    
    # Visualize the saliency map
    plt.figure(figsize=(10, 5))
    
    # Plot the original input frame
    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor[0, target_frame_idx].permute(1, 2, 0).detach().numpy())
    plt.title(f'Original Frame {target_frame_idx}')
    plt.axis('off')
    
    # Plot the saliency map
    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map.detach().numpy(), cmap='hot')
    plt.colorbar()
    plt.title(f'Saliency Map for Frame {target_frame_idx}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics about the saliency map
    print(f"Saliency map shape: {saliency_map.shape}")
    print(f"Saliency map min value: {saliency_map.min().item()}")
    print(f"Saliency map max value: {saliency_map.max().item()}")
    print(f"Saliency map mean value: {saliency_map.mean().item()}")