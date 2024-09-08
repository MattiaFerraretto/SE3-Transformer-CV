import argparse
from DataLoader import FaceLandmarkDataset
from se3.model import SE3Unet

import yaml
import wandb

from trainer import train_loop

parser = argparse.ArgumentParser(description='Training configuration')

parser.add_argument('--config_path', type=str, help='Training configuration file', required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config_path) as fp:
        conf = yaml.safe_load(fp)

    train_set = FaceLandmarkDataset(**conf['train_set'])
    test_set = FaceLandmarkDataset(**conf['test_set'])
    model = SE3Unet(**conf['model'])

    wandb.login(key=conf['wandb']['api'])

    wandb.init(
        project=conf['wandb']['project'],
        name=conf['wandb']['run_name'],
        config=conf['hyper']
    )


    train_loop(
        model=model,
        train_set=train_set,
        eval_set=test_set,
        config=conf['hyper']
    )