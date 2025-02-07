from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def augmentation(faces, landmark_gts, transformation=None):
   # if transformation == "none":
   #     return faces, landmark_gts
    
    # Load inputs on GPU
    faces = faces.to(device)
    landmark_gts = landmark_gts.to(device)
    ### Random rotation ###
    random_rotation = Rotation.random(faces.shape[0])
    rotation_matrix = torch.tensor(random_rotation.as_matrix(), dtype=torch.float32, device=device).permute(0,2,1)
    ### Random translation ### 
    translation_vector = torch.rand(faces.shape[0], dtype=torch.float32).to(device) * 10
    faces_rotaed_tensor = torch.zeros(faces.shape)
    landmark_gts_rotated_tensor = torch.zeros(landmark_gts.shape)
    for i in range(faces.shape[0]):
        if transformation == "rotation":
            faces_new = torch.tensordot(faces[i], rotation_matrix[i], dims=([1], [0]))
            landmark_gts_new = torch.tensordot(landmark_gts[i], rotation_matrix[i], dims=([1], [0]))
            # add transformed data
            faces_rotaed_tensor[i] = faces_new
            landmark_gts_rotated_tensor[i] = landmark_gts_new
        if transformation == "translation":
            faces_new = faces[i] + translation_vector[i].repeat(faces[i].shape)
            landmark_gts_new = landmark_gts[i] + translation_vector[i].repeat(landmark_gts[i].shape)
            # add transformed data
            faces_rotaed_tensor[i] = faces_new
            landmark_gts_rotated_tensor[i] = landmark_gts_new
        if transformation == "rotation_translation":
            # rotation
            faces_new = torch.matmul(faces[i], rotation_matrix[i])
            landmark_gts_new = torch.matmul(landmark_gts[i], rotation_matrix[i])
            # translation
            faces_new = faces_new + translation_vector[i].repeat(faces_new.shape)
            landmark_gts_new = landmark_gts_new + translation_vector[i].repeat(landmark_gts_new.shape)
            # add transformed data
            faces_rotaed_tensor[i] = faces_new
            landmark_gts_rotated_tensor[i] = landmark_gts_new
        
    return faces_rotaed_tensor, landmark_gts_rotated_tensor
