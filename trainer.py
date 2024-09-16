import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os
from tqdm import trange

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

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        return F_loss.mean()

def compute_pos_weight(labels: torch.tensor, threshold: float=0.5):
    labels = labels.reshape(-1, labels.shape[-1])
    H, _ = labels.shape

    return H / (labels > threshold).sum(dim=0)

def eval_loop(model: nn.Module, eval_set: Dataset, criterion: WBCEWithLogits, device, features, batch_size):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in trange(0, len(eval_set), batch_size, desc="Evaluating.."):
            pointclouds, y = eval_set[i : i + batch_size]
            
            y_hat = model(
                pointclouds.to(device),
                features,
                batch_size
            )
            
            loss = criterion(
                y_hat,
                y.to(device)
            )
            
            total_loss += loss.item()

    return total_loss / len(eval_set)

def save_checkpoint(model: nn.Module, optimizer: Adam, epoch: int, val_loss: float, out_dir: str, save_every: int,  save_max: int):
    if (epoch + 1) % save_every != 0:
      return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    path = os.path.join(out_dir, f'checkpoint-{epoch+1}.pth')
    torch.save(checkpoint, path)

    checkpoints = [f for f in os.listdir(out_dir) if f.startswith('checkpoint-')]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    while len(checkpoints) > save_max:
        os.remove(os.path.join(out_dir, checkpoints.pop(0)))



def train_loop(model: nn.Module, train_set: Dataset, eval_set: Dataset, config):

    if config['from_checkpoint']:
        checkpoint = torch.load(config['checkpoint_fpath'], map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6, last_epoch=checkpoint['epoch'])
        
        start_epoch = checkpoint['epoch'] + 1
    else:
        #optimizer = SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
        optimizer = Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
        start_epoch = 0
    
    model.to(config['device'])

    #_, labels = train_set[:]
    pos_weight = compute_pos_weight(train_set.heatmaps)
    
    #criterion = WBCEWithLogits()
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config['device']))
    criterion = FocalLoss(pos_weight=pos_weight.to(config['device']))

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        running_loss = 0.0

        for i in trange(0, len(train_set), config['batch_training_size'],  desc=f"Epoch {epoch+1}/{config['epochs']}"):
            pointclouds, y = train_set[i : i + config['batch_training_size']]

            y_hat = model(
                pointclouds.to(config['device']),
                config['features'],
                config['batch_training_size']
            )

            loss = criterion(
                y_hat,
                y.to(config['device'])
            )

            loss = loss / config['gradient_accumulation_steps']

            loss.backward()

            if (i + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * config['gradient_accumulation_steps']

            if i % config['logging_steps'] == 0:
                wandb.log({
                    "train_loss": loss.item() * config['gradient_accumulation_steps']
                })
        
        if (i + 1) % config['gradient_accumulation_steps'] != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_set)

        val_loss = eval_loop(
            model,
            eval_set,
            criterion,
            config['device'],
            config['features'],
            config['batch_eval_size']
        )

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        scheduler.step()

        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            config['checkpoint_dir'],
            config['save_every'],
            config['save_max']
        )

    wandb.finish()