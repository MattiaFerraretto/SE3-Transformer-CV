import argparse
from DataLoader import FaceLandmarkDataset
from se3.model import SE3Unet
import yaml
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import trange
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Test configuration')

parser.add_argument('--config_path', type=str, help='Test configuration file', required=True)


def compute_scores(y, y_hat):

    thresholds = torch.arange(0, 1, 0.01)
    #arg_threshold = torch.where(thresholds == 0.5)[0].item()

    precisions = torch.zeros(y.shape[0], thresholds.shape[0])
    recalls = torch.zeros(y.shape[0], thresholds.shape[0])
    f1_scores = torch.zeros(y.shape[0], thresholds.shape[0])


    for i, threshold in enumerate(thresholds):
        y_bin_hat = (torch.sigmoid(y_hat) > threshold).float()

        tp = ((y == 1) & (y_bin_hat == 1)).sum(axis=-2).sum(-1)
        fn = ((y == 1) & (y_bin_hat == 0)).sum(axis=-2).sum(-1)
        fp = ((y == 0) & (y_bin_hat == 1)).sum(axis=-2).sum(-1)
        tn = ((y == 0) & (y_bin_hat == 0)).sum(axis=-2).sum(-1)

        # tp = ((y == 1) & (y_bin_hat == 1)).sum(axis=-2)
        # fn = ((y == 1) & (y_bin_hat == 0)).sum(axis=-2)
        # fp = ((y == 0) & (y_bin_hat == 1)).sum(axis=-2)
        # tn = ((y == 0) & (y_bin_hat == 0)).sum(axis=-2)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn )
        f1_score = 2 * tp / (2 * tp + fp + fn)

        precisions[:, i] = precision
        recalls[:, i] = recall
        f1_scores[:, i] = f1_score


        # precisions[:, i] = precision.mean(axis=-1)
        # recalls[:, i] = recall.mean(axis=-1)
        # f1_scores[:, i] = f1_score.mean(axis=-1)

    precisions = torch.nan_to_num(precisions)
    recalls = torch.nan_to_num(recalls)
    f1_scores = torch.nan_to_num(f1_scores)

    return thresholds, precisions, recalls, f1_scores

def plot_metrics(thresholds, precision, recall, f1_scores, save=True):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))
    plt.style.use('seaborn-v0_8-darkgrid')

    def style_subplot(ax, title):
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.tick_params(labelsize=10)

    # 1. Precision Curve
    ax1.plot(thresholds, precision, color='#2E86C1', lw=2)
    ax1.set_xlabel('Threshold', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=10, fontweight='bold')
    style_subplot(ax1, 'Precision Curve')

    # 2. Recall Curve
    ax2.plot(thresholds, recall, color='#27AE60', lw=2)
    ax2.set_xlabel('Threshold', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Recall', fontsize=10, fontweight='bold')
    style_subplot(ax2, 'Recall Curve')

    # 3. F1 Score Curve
    ax3.plot(thresholds, f1_scores, color='#8E44AD', lw=2)
    ax3.set_xlabel('Threshold', fontsize=10, fontweight='bold')
    ax3.set_ylabel('F1 Score', fontsize=10, fontweight='bold')
    style_subplot(ax3, 'F1 Score Curve')

    # 4. Precision-Recall Curve
    ax4.plot(recall, precision, color='#E67E22', lw=2)

    ax4.set_xlabel('Recall', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=10, fontweight='bold')
    style_subplot(ax4, 'Precision-Recall Curve')

    fig.suptitle('Classification Metrics Analysis', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save:
        plt.savefig('classification_metrics.png', dpi=300, bbox_inches='tight')

    plt.show()

def test_loop(model: nn.Module, test_set: Dataset, batch_size:int, features: str, device: str):
    model.eval()

    precisions = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        for i in trange(0, len(test_set), batch_size, desc="Testing.."):
            pointclouds, y = test_set[i : i + batch_size]
            
            y_hat = model(
                pointclouds.to(device),
                features,
                y.shape[0]
            )

            thresholds, precision_per_elem, recall_per_elem, f1_score_per_elem = compute_scores(y, y_hat)
            
            precisions.append(precision_per_elem)
            recalls.append(recall_per_elem)
            f1_scores.append(f1_score_per_elem)

            if i >= 15:
                break

    precisions = torch.cat(precisions, axis=0).mean(axis=0)
    recalls = torch.cat(recalls, axis=0).mean(axis=0)
    f1_scores = torch.cat(f1_scores, axis=0).mean(axis=0)

    plot_metrics(thresholds, precisions, recalls, f1_scores)



if __name__ == "__main__":
    args = parser.parse_args()

    with open('./test-conf-example.yaml') as fp:
        conf = yaml.safe_load(fp)

    test_set = FaceLandmarkDataset(**conf['test_set'])
    model = SE3Unet(**conf['model'])

    checkpoint = torch.load(conf['checkpoint_fpath'], map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    device = conf['hyper']['device']
    batch_size = conf['hyper']['batch_size']
    features = conf['hyper']['features']

    model.to(device)

    test_loop(model, test_set, batch_size, features, device)
    