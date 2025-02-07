import argparse
from DataLoader import FaceLandmarkDataset
from se3.model import SE3Unet, SE3UnetV2
import yaml
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import trange
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import defaultdict
import os

parser = argparse.ArgumentParser(description='Test configuration')

parser.add_argument('--config_path', type=str, help='Test configuration file', required=True)

def compute_IoU(y, y_hat, threshold=0.2):
    intersection = ((y_hat >= threshold) & (y >= threshold)).sum(axis=1)
    union = ((y_hat >= threshold) | (y >= threshold)).sum(axis=1)

    return torch.nan_to_num(intersection / union,  nan=0.0)

def plot_IoUs(thresolds, IoUs, title, save_to):
    num_subplots = 68
    rows, cols = 10, 7

    plt.style.use("seaborn-v0_8-muted")

    fig, axes = plt.subplots(rows, cols, figsize=(15, 20), constrained_layout=True)
    axes = axes.flatten()

    for i in range(num_subplots):
        axes[i].plot(thresolds, IoUs[:, i],linewidth=1.5, color="tab:blue")
        
        axes[i].grid(True, linestyle="--", alpha=0.6)
        axes[i].set_title(f"#{i}", fontsize=8, weight="bold")
        axes[i].tick_params(axis="both", which="major", labelsize=7)

        axes[i].set_xlim(0.1, 1)
        axes[i].set_ylim(0, 1)

    for i in range(num_subplots, len(axes)):
        fig.delaxes(axes[i])
    
    
    plt.savefig(os.path.join(save_to, f"{title}.png"), bbox_inches="tight", dpi=300)
    plt.close()

def test_loop(model: nn.Module, test_set: Dataset, batch_size:int, features: str, device: str, conf: dict):
    model.eval()

    IoUs = []
    thresolds = torch.arange(0.1, 1, 0.05)
    IoUs = defaultdict(list)

    with torch.no_grad():
        for i in trange(0, len(test_set), batch_size, desc="Testing.."):
            pointclouds, y = test_set[i : i + batch_size]
            
            y_hat = model(
                pointclouds.to(device),
                features,
                y.shape[0]
            )

            for t in thresolds:
                IoU = compute_IoU(y, y_hat, t)
                IoUs[f"{t.item():.2f}"].append(IoU)


    avg_IoUs = []
    for result in IoUs.values():
        avg_IoUs.append(torch.cat(result, axis=0).mean(axis=0, keepdim=True))

    avg_IoUs = torch.cat(avg_IoUs, axis=0)

    avg_IoU = torch.cat(IoUs["0.20"], axis=0).mean(axis=0)
    table_result = tabulate([[f"{i}", f"{v:.4f}"] for i, v in enumerate(avg_IoU)], headers=["#", "IoU score"], tablefmt="presto")
    output = (
        "# Test results\n\n"
        f"{table_result}\n\n"
        f"argmin: {avg_IoU.argmin()}, min: {avg_IoU.min():.4f}\n"
        f"argmax: {avg_IoU.argmax()}, max: {avg_IoU.max():.4f}\n"
    )

    os.makedirs(conf['results_dir'], exist_ok=True)

    with open(
        os.path.join(
            conf['results_dir'],
            f"test_{conf['test_set']['preprocessing']}_{conf['test_set']['break_ds_with']}.txt"
        ),
        'w'
    ) as fp:
        fp.write(output)
        fp.flush()

    plot_IoUs(
        thresolds,
        avg_IoUs,
        title=f"test_{conf['test_set']['preprocessing']}_{conf['test_set']['break_ds_with']}",
        save_to=conf['results_dir']
    )



if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config_path) as fp:
        conf = yaml.safe_load(fp)

    test_set = FaceLandmarkDataset(**conf['test_set'])
    model = SE3UnetV2(**conf['model'])

    checkpoint = torch.load(conf['checkpoint_fpath'], map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    device = conf['hyper']['device']
    batch_size = conf['hyper']['batch_size']
    features = conf['hyper']['features']

    model.to(device)

    test_loop(model, test_set, batch_size, features, device, conf)
    