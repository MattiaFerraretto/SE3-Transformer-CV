import os
import torch
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data

from Preprocessing.augmentation import augmentation
from Preprocessing.procrustes_icp import batch_icp

import dgl
from dgl.geometry import farthest_point_sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceLandmarkDataset(Dataset):
    def __init__(
            self,
            preprocessing,
            break_ds_with,
            ds_path,
            split,
            category,
            references_pointclouds_icp_path,
            reduce_pointcloud_to: int=None
        ):
        print("Loading dataset...")

        self.preprocessing = preprocessing
        self.break_ds_with = break_ds_with
        self.split = split
        self.ds_path = os.path.join(ds_path, split, f"{split.lower()}_neutral.npy")
        self.ds_name = self.ds_path.split("/")[1]
        self.category = category
        self.reduce_pointcloud_to = reduce_pointcloud_to
        
        self.faces, self.landmark_gts, self.heatmaps, self.scales = self.load_face_data()

        if self.reduce_pointcloud_to != 'None':
            self.faces, self.heatmaps = self._reduce(
                n_points=self.reduce_pointcloud_to,
                point_cloud=self.faces,
                landmarks=self.landmark_gts,
                sigma=0.095
            )
            
        # if it is not none, then i want it aligned.
        # However, if I break it with an E(3) transformation, then I need a pre-processing step for the model.
        if self.preprocessing == 'icp' and self.break_ds_with != 'none':
            pointcloud_ref, landmarks_ref = self.save_pt_reference_for_icp(
                self.faces[0],
                self.landmark_gts[0],
                references_pointclouds_icp_path,
                os.path.join(
                    references_pointclouds_icp_path,
                    self.ds_name+"_"+self.category+"_"+self.split+"_reference_pointcloud_icp.pt"
                )
            )
            self.faces_aug, self.landmark_gts_aug = augmentation(self.faces, self.landmark_gts, transformation=self.break_ds_with)
            self.faces, self.landmark_gts = batch_icp(self.faces_aug, self.landmark_gts_aug, pointcloud_ref, landmarks_ref)
            
        elif self.preprocessing == 'spatial_transformer' and self.break_ds_with != 'none':
            self.faces, self.landmark_gts = augmentation(self.faces, self.landmark_gts, transformation=self.break_ds_with)
        

        print("Preprocessing:", self.preprocessing)
        print("Dataset:", self.ds_name)
        print("Category:", self.category)
        print("\n")

        print("Len dataset: ", self.faces.shape)
        print("Face: ", self.faces.shape)
        print("Landmark: ", self.landmark_gts.shape)
        print("Heatmaps: ", self.heatmaps.shape)
        #print("Filename: ", self.filenames.shape)
        print("Scale: ", self.scales.shape)
        # if self.ds_name != 'York' and self.ds_name != 'Lafas': 
        #     print("Emotion: ", set(emotions))
        
    def save_pt_reference_for_icp(self, pointcloud, landmarks, folder, path):
        os.makedirs(folder, exist_ok = True) 
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
        
        return torch.tensor(faces,  dtype=torch.float32), torch.tensor(landmark_gts,  dtype=torch.float32), torch.tensor(heatmaps,  dtype=torch.float32), torch.tensor(scales,  dtype=torch.float32)
    
    
    def _reduce(self, n_points: int, point_cloud: torch.tensor,landmarks: torch.tensor, sigma: float=0.08):
        if point_cloud.dim() == 2:
            point_cloud = torch.unsqueeze(point_cloud, dim=0)  
            landmarks = torch.unsqueeze(landmarks, dim=0)

        
        centroids = torch.mean(point_cloud, axis=-2)
        avg_centroids = torch.mean(centroids, axis=0)
        fp_start_idx = torch.argmin(torch.mean(torch.linalg.norm(point_cloud - avg_centroids[None, None, :], dim=-1), axis=0))

        point_idxs = farthest_point_sampler(point_cloud, n_points, fp_start_idx)
        dim0_index = torch.arange(point_cloud.shape[0]).unsqueeze(-1)

        point_cloud = point_cloud[dim0_index, point_idxs]

        sqrd_distances = torch.sum((point_cloud[:, :, None, :] - landmarks[:, None, :, :])**2, dim=-1)
        heatmap = torch.exp(- sqrd_distances / (2 * sigma ** 2))

        return point_cloud, heatmap

    def __getitem__(self, item):
        
        #landmark = self.landmark_gts[item]
        #scale = self.scales[item]
        #if self.reduce_pointcloud_to != 'None':
        #    face, heatmap = self._reduce(
        #        n_points=self.reduce_pointcloud_to,
        #        point_cloud=self.faces[item],
        #        landmarks=self.landmark_gts[item]
        #    )
        #else:
        face = self.faces[item]
        heatmap = self.heatmaps[item] if self.heatmaps[item].dim() == 3 else torch.unsqueeze(self.heatmaps[item], dim=0)

        knn_g = dgl.knn_graph(face, k=40, algorithm='kd-tree', exclude_self=True)
        
        indices_src, indices_dst = knn_g.edges()
 
        face = face.view(-1, 3)
        knn_g.ndata['x'] = torch.unsqueeze(face, dim=1)
        knn_g.ndata['v'] = torch.unsqueeze(face, dim=1)
        knn_g.edata['d'] = face[indices_dst] - face[indices_src]
        knn_g.edata['w'] = torch.sqrt(torch.sum(knn_g.edata['d']**2, dim=-1, keepdim=True))

        return knn_g, heatmap #torch.clamp(heatmap, min=1e-8) (heatmap > 0.5).float()

    def __len__(self):
        return self.faces.shape[0]