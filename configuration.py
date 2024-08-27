import argparse

# resampled datsets
path_datasets_resampled = "./MattiaFerrarettoDataset"
#path_saving_references_pointclouds_icp = "./MattiaFerrarettoDataset/Preprocessing/reference_pointclouds_for_icp"
path_saving_references_pointclouds_icp = "./Preprocessing/reference_pointclouds_for_icp"

parser = argparse.ArgumentParser(description='3D Face Landmark Detection')
# base args
parser.add_argument('--preprocessing', type=str, default='none', help='Preprocess the dataset', choices=['icp', 'spatial_transformer', 'none'])
parser.add_argument('--break_ds_with', type=str, default='rotation_translation', help='Preprocess the dataset', choices=['rotation', 'translation', 'rotation_translation', 'none'])
parser.add_argument('--dataset', type=str, default='Facescape', metavar='N', choices=['Facescape'])
parser.add_argument('--category', type=str, default='Neutral', metavar='N', choices=['Neutral'])
