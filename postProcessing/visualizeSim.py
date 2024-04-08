import torch
import matplotlib.pyplot as plt
import os
import tqdm

def visualize():
    directory = "visualize"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fileName = "segbev.pt"
    loaded_tensors = torch.load(fileName)
    for i in tqdm.tqdm(range(len(loaded_tensors))):
        innerD = os.path.join(directory, f"Timestep{i}")
        if not os.path.exists(innerD):
            os.makedirs(innerD)
        for j in range(4):
            plt.imshow(loaded_tensors[i][0,j].cpu().detach().numpy(), cmap='gray')
            plt.colorbar()

            filename = os.path.join(innerD, f"Channel{j}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.clf()

if __name__ == "__main__":
    visualize()
