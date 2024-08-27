import os
import torch
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data

from configuration import *
from Preprocessing.augmentation import augmentation
from Preprocessing.procrustes_icp import batch_icp

import dgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceLandmarkDataset(Dataset):
    def __init__(
            self,
            preprocessing,
            break_ds_with,
            ds_path,
            split,
            category,
            references_pointclouds_icp_path
        ):
        print("Loading dataset...")

        self.preprocessing = preprocessing
        self.break_ds_with = break_ds_with
        self.split = split
        self.ds_path = os.path.join(ds_path, split, f"{split.lower()}_neutral.npy")
        self.ds_name = self.ds_path.split("/")[1]
        self.category = category
        
        self.faces, self.landmark_gts, self.heatmaps, self.scales = self.load_face_data()
            
        # if it is not none, then i want it aligned.
        # However, if I break it with an E(3) transformation, then I need a pre-processing step for the model.
        if self.preprocessing == 'icp' and self.break_ds_with != 'none':
            pointcloud_ref, landmarks_ref = self.save_pt_reference_for_icp(self.faces[0], self.landmark_gts[0], os.path.join(references_pointclouds_icp_path, self.ds_name+"_"+self.category+"_reference_pointcloud_icp.pt"))
            self.faces_aug, self.landmark_gts_aug = augmentation(self.faces, self.landmark_gts, transformation=self.break_ds_with)
            self.faces, self.landmark_gts = batch_icp(self.faces_aug, self.landmark_gts_aug, pointcloud_ref, landmarks_ref)
        elif self.preprocessing == 'spatial_transformer' and self.break_ds_with != 'none':
            self.faces, self.landmark_gts = augmentation(self.faces, self.landmark_gts, transformation=self.break_ds_with)
        
    def save_pt_reference_for_icp(self, pointcloud, landmarks, path):
        if not os.path.exists(path):
            torch.save(Data(data=pointcloud, landmarks=landmarks), path)
        return torch.load(path).data, torch.load(path).landmarks
        
    def load_face_data(self):
        assert self.ds_path is not None, "Path to dataset is not provided"
        print("Path file: ", self.ds_path)
        dataset = np.load(self.ds_path, allow_pickle=True)
        faces = dataset['point_list']
        landmark_gts = dataset['landmark_list']
        filenames = dataset['subject_list']
        heatmaps = dataset['heatmaps_list']
        if self.ds_name != 'York' and self.ds_name != 'Lafas':
            emotions = dataset['emotion_list']
        scales = dataset['scale_list']


        print("Preprocessing:", self.preprocessing)
        print("Dataset:", self.ds_name)
        print("Category:", self.category)
        print("\n")

        print("Len dataset: ", faces.shape)
        print("Face: ", faces.shape)
        print("Landmark: ", landmark_gts.shape)
        print("Heatmaps: ", heatmaps.shape)
        print("Filename: ", filenames.shape)
        print("Scale: ", scales.shape)
        if self.ds_name != 'York' and self.ds_name != 'Lafas': 
            print("Emotion: ", set(emotions))
        
        return torch.tensor(faces,  dtype=torch.float32), torch.tensor(landmark_gts,  dtype=torch.float32), torch.tensor(heatmaps,  dtype=torch.float32), torch.tensor(scales,  dtype=torch.float32)

    def __getitem__(self, item):
        face = self.faces[item]
        #landmark = self.landmark_gts[item]
        heatmap = self.heatmaps[item]
        #scale = self.scales[item]

        #knn_g = dgl.knn_graph(self.faces[item], k=10, algorithm='kd-tree', exclude_self=True)
        knn_g = dgl.knn_graph(face, k=40, algorithm='kd-tree', exclude_self=True)
        knn_g.k = 10
        
        indices_src, indices_dst = knn_g.edges()
 
        face = face.view(-1, 3)
        knn_g.ndata['x'] = torch.unsqueeze(face, dim=1)
        knn_g.ndata['v'] = torch.unsqueeze(face, dim=1)
        knn_g.edata['d'] = face[indices_dst] - face[indices_src]
        knn_g.edata['w'] = torch.sqrt(torch.sum(knn_g.edata['d']**2, dim=-1, keepdim=True))

        return knn_g, heatmap
        #return {'face': face, 'landmark': landmark, 'heatmap':heatmap, 'scale':scale}

    def __len__(self):
        return self.faces.shape[0]