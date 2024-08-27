
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

import plotly.express as px
import plotly.graph_objects as go

from scipy.linalg import orthogonal_procrustes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def procrustes(data1, data2):
  
    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) #* s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return R, disparity

# Function to convert numpy array to open3d point cloud
def numpy_to_o3d_point_cloud(np_array):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np_array)
    return point_cloud

def visualize_pointcloud(pointcloud):
    fig = px.scatter_3d(x=pointcloud[:,0], y=pointcloud[:,1], z=pointcloud[:,2], width=1280, height=720)
    fig.update_traces(marker=dict(size=2, color='black'))
    center_of_mass = torch.mean(pointcloud.unsqueeze(0), dim=1, keepdim=True).squeeze(1)
    fig.add_trace(go.Scatter3d(x=center_of_mass[:, 0], y=center_of_mass[:, 1], z=center_of_mass[:, 2], mode='markers', marker=dict(size=5, color='red')))
    fig1 = go.Figure(data=fig.data)
    fig1.show()
    
def visualize_two_pointclouds(pointcloud1, pointcloud2):
    fig = px.scatter_3d(x=pointcloud1[:,0], y=pointcloud1[:,1], z=pointcloud1[:,2], width=1280, height=720)
    fig.update_traces(marker=dict(size=2, color='blue'), name='PointCloud1')
    
    fig.add_trace(go.Scatter3d(x=pointcloud2[:,0], y=pointcloud2[:,1], z=pointcloud2[:,2], mode='markers', marker=dict(size=2, color='green'), name='PointCloud2'))

    center_of_mass1 = torch.mean(pointcloud1.unsqueeze(0), dim=1, keepdim=True).squeeze(1)
    center_of_mass2 = torch.mean(pointcloud2.unsqueeze(0), dim=1, keepdim=True).squeeze(1)

    fig.add_trace(go.Scatter3d(x=center_of_mass1[:, 0], y=center_of_mass1[:, 1], z=center_of_mass1[:, 2], mode='markers', marker=dict(size=5, color='red'), name='CenterOfMass1'))
    fig.add_trace(go.Scatter3d(x=center_of_mass2[:, 0], y=center_of_mass2[:, 1], z=center_of_mass2[:, 2], mode='markers', marker=dict(size=5, color='orange'), name='CenterOfMass2'))
    
    fig.show()
    
def visualize_pointclouds_aligned(pointcloud1, pointcloud2):
    # Calculate centers of mass
    center_of_mass1 = torch.mean(pointcloud1.unsqueeze(0), dim=1, keepdim=True).squeeze(1)
    center_of_mass2 = torch.mean(pointcloud2.unsqueeze(0), dim=1, keepdim=True).squeeze(1)
    
    # Translate pointcloud2 to align its center of mass with pointcloud1's center of mass
    translation_vector = center_of_mass1 - center_of_mass2
    pointcloud2_aligned = pointcloud2 + translation_vector

    # Create the plot
    fig = px.scatter_3d(x=pointcloud1[:,0], y=pointcloud1[:,1], z=pointcloud1[:,2], width=1280, height=720)
    fig.update_traces(marker=dict(size=2, color='blue'), name='PointCloud1')
    
    fig.add_trace(go.Scatter3d(x=pointcloud2_aligned[:,0], y=pointcloud2_aligned[:,1], z=pointcloud2_aligned[:,2], mode='markers', marker=dict(size=2, color='green'), name='PointCloud2'))

    fig.add_trace(go.Scatter3d(x=center_of_mass1[:, 0], y=center_of_mass1[:, 1], z=center_of_mass1[:, 2], mode='markers', marker=dict(size=5, color='red'), name='CenterOfMass1'))
    fig.add_trace(go.Scatter3d(x=center_of_mass1[:, 0], y=center_of_mass1[:, 1], z=center_of_mass1[:, 2], mode='markers', marker=dict(size=5, color='orange'), name='CenterOfMass2Aligned'))
    
    fig.show()
    
# Function to perform ICP on a batch of point clouds
def batch_icp(batch_point_clouds, batch_landmark_gts, reference_point_cloud, reference_landmarks, threshold=0.02, max_iteration=2000):
    # Ensure input tensors are on the same device
    device = batch_point_clouds.device
    batch_landmark_gts = batch_landmark_gts.to(device)

    # Concatenate tensors along the second dimension (axis 1)
    batch_point_clouds_combined = torch.cat((batch_point_clouds, batch_landmark_gts), dim=1)
    reference_combined = torch.cat((reference_point_cloud, reference_landmarks), dim=0)
    
    # Convert reference point cloud to numpy and then to Open3D
    reference_pc = numpy_to_o3d_point_cloud(reference_point_cloud.cpu().numpy())
    #visualize_pointcloud(reference_point_cloud.cpu()) # REFERENCE POINT CLOUD

    # Prepare output tensors
    B, N, _ = batch_point_clouds.shape
    L = batch_landmark_gts.shape[1]
    aligned_point_clouds_tensor = torch.zeros((B, N, 3), dtype=torch.float32, device=device)
    aligned_landmark_gts_tensor = torch.zeros((B, L, 3), dtype=torch.float32, device=device)

    for i in tqdm(range(batch_point_clouds_combined.shape[0]), desc="Aligning..."):
        # ALIGN POINT CLOUDS
        center_of_mass1 = torch.mean(batch_point_clouds_combined[i].unsqueeze(0), dim=1, keepdim=True).squeeze(1)
        batch_point_clouds_combined[i] = batch_point_clouds_combined[i] - center_of_mass1
        center_of_mass2 = torch.mean(reference_point_cloud.unsqueeze(0), dim=1, keepdim=True).squeeze(1)
        batch_point_clouds_combined[i] = batch_point_clouds_combined[i] + center_of_mass2
        
        # PROCUSTES ALIGNMENT
        R, _ = procrustes(batch_point_clouds_combined[i,N:].cpu().numpy(), reference_combined[N:].cpu().numpy())
        batch_point_clouds_combined_aligned = torch.tensor(np.dot(np.array(batch_point_clouds_combined[i].cpu().numpy(), dtype=np.float64, copy=True), R))
        
        # CHECKING ALIGNMENT 
        #visualize_two_pointclouds(batch_point_clouds_combined[i], reference_point_cloud) # POINT CLOUDS NOT ALIGNED
        #visualize_two_pointclouds(torch.tensor(batch_point_clouds_combined_aligned[N:]), reference_landmarks) # LANDMARKS
        #visualize_two_pointclouds(torch.tensor(batch_point_clouds_combined_aligned[:N]), reference_point_cloud) # POINT CLOUDS ALIGNED
    
        # ICP
        source_pc = numpy_to_o3d_point_cloud((batch_point_clouds_combined_aligned).cpu().numpy())
        result_icp = o3d.pipelines.registration.registration_icp(
            source_pc, reference_pc, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
        )

        transformation = result_icp.transformation
        source_pc.transform(transformation)
        #visualize_two_pointclouds(torch.tensor(np.asarray(source_pc.points)), reference_point_cloud)

        # Split aligned point cloud and landmarks
        aligned_pc_np = np.asarray(source_pc.points)
        aligned_point_clouds_tensor[i] = torch.tensor(aligned_pc_np[:N], dtype=torch.float32, device=device)
        aligned_landmark_gts_tensor[i] = torch.tensor(aligned_pc_np[N:], dtype=torch.float32, device=device)

    return aligned_point_clouds_tensor, aligned_landmark_gts_tensor