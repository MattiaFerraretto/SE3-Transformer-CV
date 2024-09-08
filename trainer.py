import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os
from tqdm import tqdm

def eval_loop(model: nn.Module, eval_set: Dataset, criterion: nn.BCEWithLogitsLoss, device, features, batch_size):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for pointclouds, y in eval_set:
            
            y_hat = model(
                pointclouds.to(device),
                features,
                batch_size
            )
            loss = criterion(y_hat, y)
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
    
    model.to(config['device'])
    #optimizer = SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    scheduler = CosineAnnealingLR(optimizer, config['epochs'], eta_min=0.001)

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0

        for i, entry in enumerate(tqdm(train_set, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            pointclouds, y = entry[0], entry[1]

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