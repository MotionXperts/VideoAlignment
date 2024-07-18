import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import torch

def create_animation(X_norm,frame_idx):
    split_anchor = np.where(frame_idx == 0)[0][1]
    frame_idcies = [frame_idx[:split_anchor], frame_idx[split_anchor:]]
    X_norms = [X_norm[:split_anchor], X_norm[split_anchor:]]
    

    # Initialize the plot
    fig = plt.figure(figsize=(8, 8))

    # Lists to store all points
    annotations = []

    # Initialization function
    def init():
        plt.xticks([])
        plt.yticks([])
        for annotation in annotations:
            annotation.remove()
        annotations.clear()
        return []

    # Animation function
    def animate(i):
        plt.xticks([])
        plt.yticks([])
        # Generate two new points
        if i < len(frame_idcies[0]):
            x1, y1 = X_norms[0][i]
            annotation1 = plt.text(x1, y1, frame_idcies[0][i], fontsize=12, color='blue')
            annotations.append(annotation1)
        if i < len(frame_idcies[1]):
            x2, y2 = X_norms[1][i]
            annotation2 = plt.text(x2, y2, frame_idcies[1][i], fontsize=12, color='red')
            annotations.append(annotation2)

        return annotations

    # Create animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=max(len(frame_idcies[0]),len(frame_idcies[1])), interval=300, blit=True)

    output_file = 'animation.mp4'
    ani.save(output_file, dpi=80)

    plt.close(fig)  # Close the figure to avoid displaying it in tests
    return output_file

if __name__ == "__main__":
    X_norm = torch.load('X_norm.pt')
    frame_idx = torch.load('frame_idx.pt')
    create_animation(X_norm,frame_idx)
