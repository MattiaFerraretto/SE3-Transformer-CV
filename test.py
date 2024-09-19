import argparse
from DataLoader import FaceLandmarkDataset
from se3.model import SE3Unet

import yaml

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from tqdm import trange

from sklearn.metrics import multilabel_confusion_matrix

parser = argparse.ArgumentParser(description='Test configuration')

parser.add_argument('--config_path', type=str, help='Test configuration file', required=True)

class WBCEWithLogits(nn.Module):
    def __init__(self, threshold: float=0.5, max_clamp: float = 200.0):
        super(WBCEWithLogits, self).__init__()
        self.threshold = threshold
        self.max_clamp = max_clamp

    
    def forward(self, logits, targets):
        pos_weight = (targets.shape[0] / (targets > self.threshold).sum(dim=0)).clamp(1, self.max_clamp)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='mean'
        )

        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )

        print(BCE_loss.max())
        pt = torch.exp(-BCE_loss)
        print(pt.max())
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        return F_loss.sum()


def compute_pos_weight(labels: torch.tensor, threshold: float=0.5):
    labels = labels.reshape(-1, labels.shape[-1])
    H, _ = labels.shape

    return H / (labels > threshold).sum(dim=0) 

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config_path) as fp:
        conf = yaml.safe_load(fp)

    
    test_set = FaceLandmarkDataset(**conf['test_set'])
    model = SE3Unet(**conf['model'])

    checkpoint = torch.load(conf['checkpoint_fpath'], map_location='cpu')
    
    #model.load_state_dict(checkpoint['model_state_dict'])

    device = conf['hyper']['device']
    batch_size = conf['hyper']['batch_size']
    features = conf['hyper']['features']

    model.to(device)
    model.eval()

    pos_weigth = compute_pos_weight(test_set.heatmaps)
    print(pos_weigth)
    criterion = FocalLoss(alpha=0.085, gamma=2, pos_weight= pos_weigth)

    with torch.no_grad():
        # for i in trange(0, len(test_set), batch_size, desc="Testing.."):
        #     pointclouds, y = test_set[i : i + batch_size]
            
        #     y_hat = model(
        #         pointclouds.to(device),
        #         features,
        #         batch_size
        #     )

        pointclouds, y = test_set[23]
        y_hat = model(
            pointclouds.to(device),
            features,
            batch_size
        )

        loss = criterion(
            y_hat,
            y.to(device)
        )

        print('loss:', loss.item())

        print(y_hat)
        #maxs = y_hat.argmax(dim=-1)
        #print((y > 0.5).sum())
        #print(torch.sigmoid(y_hat[0]))
        act = torch.sigmoid(y_hat[0]) > 0.5
        print((y[0] > 0.5).sum(dim=0))
        print(act.sum(dim=0))

        

