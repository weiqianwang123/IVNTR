"""Simple inference script for TableEmpty predicate.

Input: Two point cloud files (teapot.ply, table.ply)
Output: True/False indicating whether the table is empty
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from typing import List


# Import the exact same classes from the training script
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, ls_list=[64, 128, 256], global_feat=True, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(3)
        self.conv1 = torch.nn.Conv1d(3, ls_list[0], 1)
        self.conv2 = torch.nn.Conv1d(ls_list[0], ls_list[1], 1)
        self.conv3 = torch.nn.Conv1d(ls_list[1], ls_list[2], 1)
        self.bn1 = nn.BatchNorm1d(ls_list[0])
        self.bn2 = nn.BatchNorm1d(ls_list[1])
        self.bn3 = nn.BatchNorm1d(ls_list[2])
        self.final_ls = ls_list[2]
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=ls_list[0])

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.final_ls)
        
        if self.global_feat:
            return x
        else:
            x = x.view(-1, self.final_ls, 1).repeat(1, 1, N)
            return x


class TableEmptyClassifier(nn.Module):
    """TableEmpty classifier matching the training script architecture."""
    
    def __init__(self, num_points: int = 1024, pointnet_layers: List[int] = None):
        super(TableEmptyClassifier, self).__init__()
        
        if pointnet_layers is None:
            pointnet_layers = [64, 128, 256]  # Default architecture: conv layers
            
        self.num_points = num_points
        self.pointnet_layers = pointnet_layers
        
        # PointNet encoder (matching training script)
        self.pointnet = PointNetEncoder(
            pointnet_layers,
            global_feat=True,
            feature_transform=True
        )
        
        # The PointNet output dimension is the last layer (pointnet_layers[2])
        pointnet_output_dim = pointnet_layers[2]
        
        # MLP classifier head
        mlp_layers = [pointnet_output_dim, 128, 64, 1]  # Classifier layers
        self.fc1 = nn.Linear(mlp_layers[0], mlp_layers[1])
        self.fc2 = nn.Linear(mlp_layers[1], mlp_layers[2])  
        self.fc3 = nn.Linear(mlp_layers[2], mlp_layers[3])  # Binary classification
        
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(mlp_layers[1])
        self.bn2 = nn.BatchNorm1d(mlp_layers[2])
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self, x):
        """
        Args:
            x: Combined point cloud (batch_size, 3, num_points)
        Returns:
            Classification logits (batch_size, 1)
        """
        features = self.pointnet(x)
        
        x = torch.relu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def load_point_cloud(file_path: str, target_points: int = 512) -> np.ndarray:
    """Load and preprocess a point cloud file.
    
    Args:
        file_path: Path to .ply point cloud file
        target_points: Number of points to sample
        
    Returns:
        Point cloud array of shape (target_points, 3)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {file_path}")
    
    # Sample to target number of points
    if len(points) >= target_points:
        # Random sampling
        indices = np.random.choice(len(points), target_points, replace=False)
        sampled_points = points[indices]
    else:
        # Repeat points to reach target
        repeat_times = target_points // len(points) + 1
        repeated_points = np.tile(points, (repeat_times, 1))
        sampled_points = repeated_points[:target_points]
    
    return sampled_points


def predict_table_empty(teapot_pcd_path: str, 
                       table_pcd_path: str, 
                       model_path: str = None,
                       device: str = "auto") -> tuple:
    """Predict whether table is empty given teapot and table point clouds.
    
    Args:
        teapot_pcd_path: Path to teapot point cloud (.ply)
        table_pcd_path: Path to table point cloud (.ply)  
        model_path: Path to trained model (optional, will use dummy model if None)
        device: Device to use ("cpu", "cuda", or "auto")
        
    Returns:
        (prediction: bool, confidence: float)
    """
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load point clouds
    print(f"Loading teapot point cloud: {teapot_pcd_path}")
    teapot_points = load_point_cloud(teapot_pcd_path, target_points=512)
    
    print(f"Loading table point cloud: {table_pcd_path}")
    table_points = load_point_cloud(table_pcd_path, target_points=512)
    
    # Combine point clouds
    combined_points = np.vstack([teapot_points, table_points])  # Shape: (1024, 3)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(combined_points).transpose(0, 1).unsqueeze(0)  # (1, 3, 1024)
    input_tensor = input_tensor.to(device)
    
    # Load or create model
    model = TableEmptyClassifier(num_points=1024)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
    else:
        print("Warning: No trained model provided. Using randomly initialized model.")
        print("Train a model first using: python scripts/train_table_empty_predicate.py")
    
    model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
        prediction = probability > 0.5
    
    return prediction, probability


def main():
    parser = argparse.ArgumentParser(description='Predict TableEmpty from point clouds')
    parser.add_argument('teapot_pcd', help='Path to teapot point cloud (.ply)')
    parser.add_argument('table_pcd', help='Path to table point cloud (.ply)')
    parser.add_argument('--model', help='Path to trained model (.pth)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for inference')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    try:
        # Make prediction
        is_empty, confidence = predict_table_empty(
            args.teapot_pcd, 
            args.table_pcd, 
            args.model,
            args.device
        )
        
        # Output result
        if args.verbose:
            print(f"\nPrediction Details:")
            print(f"Teapot PCD: {args.teapot_pcd}")
            print(f"Table PCD: {args.table_pcd}")
            print(f"Table Empty: {is_empty}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Interpretation: {'Table is empty' if is_empty else 'Table has objects'}")
        else:
            # Simple True/False output
            print(is_empty)
        
        # Exit code: 0 for empty table, 1 for non-empty table
        sys.exit(0 if is_empty else 1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()