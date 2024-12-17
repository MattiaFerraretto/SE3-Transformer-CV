import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import wandb
import os
from tqdm import trange

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        delta_y = torch.abs(y_true - y_pred)
        
        loss_non_linear = self.omega * torch.log(1 + (delta_y / self.epsilon) ** (self.alpha - y_true))
        
        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y_true))) * (self.alpha - y_true) * ((self.theta / self.epsilon) ** (self.alpha - y_true - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - y_true))
        loss_linear = A * delta_y - C

        loss = torch.where(delta_y < self.theta, loss_non_linear, loss_linear) * (10 * (y_true >= 0.2).int() + 1)

        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, targets, inputs):

        pt = targets * torch.sigmoid(inputs) + (1 - targets) * (1 - torch.sigmoid(inputs))

        alpha = torch.where(targets == 1, self.alpha, 1 - self.alpha + 1e-8)

        F_loss = - alpha * (1 -pt) ** self.gamma * torch.log(pt + 1e-8)
        
        return F_loss.sum()

def compute_pos_weight(labels: torch.tensor, threshold: float=0.5):
    labels = labels.reshape(-1, labels.shape[-1])
    H, _ = labels.shape

    return H / (labels > threshold).sum(dim=0)

def eval_loop(model: nn.Module, eval_set: Dataset, criterion: nn.BCEWithLogitsLoss, device, features, batch_size):
    model.eval()
    eval_loss = []

    with torch.no_grad():
        for i in trange(0, len(eval_set), batch_size, desc="Evaluating.."):
            pointclouds, y = eval_set[i : i + batch_size]
            
            y_hat = model(
                pointclouds.to(device),
                features,
                y.shape[0]
            )
            
            loss = criterion(
                y.to(device),
                y_hat
            )
            
            eval_loss.append(loss.item())

    return torch.mean(torch.tensor(eval_loss, dtype=torch.float32)).item()

def save_checkpoint(model: nn.Module, optimizer: Adam, scheduler: CosineAnnealingWarmRestarts, epoch: int, val_loss: float, out_dir: str, save_every: int,  save_max: int):
    if (epoch + 1) % save_every != 0:
      return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
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
    
    model.to(config['device'])

    if config['from_checkpoint']:
        checkpoint = torch.load(config['checkpoint_fpath'], map_location=config['device'])
        
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['epochs'], eta_min=1e-8, last_epoch=checkpoint['epoch'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
    else:
        #optimizer = SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
        optimizer = Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['epochs'], eta_min=1e-8)
        start_epoch = 0

    #pos_weight = compute_pos_weight(train_set.heatmaps)
    
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config['device']))
    #criterion = FocalLoss(alpha=0.98, gamma=3)
    criterion = AdaptiveWingLoss()

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        running_loss = 0.0
        #num_iters = len(train_set)
        
        for i in trange(0, len(train_set), config['batch_training_size'],  desc=f"Epoch {epoch+1}/{config['epochs']}"):
            pointclouds, y = train_set[i : i + config['batch_training_size']]

            y_hat = model(
                pointclouds.to(config['device']),
                config['features'],
                y.shape[0]
            )

            loss = criterion(
                y.to(config['device']),
                y_hat
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
            scheduler,
            epoch,
            val_loss,
            config['checkpoint_dir'],
            config['save_every'],
            config['save_max']
        )

    wandb.finish()