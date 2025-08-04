"""Inference script for TableEmpty predicate using Point Cloud Transformer.

Input: Two point cloud files (teapot.ply, table.ply)
Output: True/False indicating whether the table is empty
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from typing import Tuple


class Point3DTransformer(nn.Module):
    """Simple 3D Point Transformer for point cloud feature extraction."""
    
    def __init__(self, embed_dim: int = 384, num_heads: int = 6, num_layers: int = 12):
        super(Point3DTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Point embedding
        self.point_embedding = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1024, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: Point cloud tensor (batch_size, num_points, 3)
            
        Returns:
            Global features (batch_size, embed_dim)
        """
        batch_size, num_points, _ = points.shape
        
        # Point embedding
        x = self.point_embedding(points)  # (batch_size, num_points, embed_dim)
        
        # Add positional encoding
        if num_points <= self.pos_encoding.size(0):
            x = x + self.pos_encoding[:num_points].unsqueeze(0)
        else:
            # Interpolate positional encoding if needed
            pos_enc = self.pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
            pos_enc = torch.nn.functional.interpolate(
                pos_enc.transpose(1, 2), size=num_points, mode='linear'
            ).transpose(1, 2)
            x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, num_points, embed_dim)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch_size, embed_dim, num_points)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, embed_dim)
        
        # Feature projection
        x = self.feature_proj(x)
        
        return x


class TableEmptyTransformerClassifier(nn.Module):
    """TableEmpty classifier using Point Cloud Transformer + MLP."""
    
    def __init__(self, 
                 num_points: int = 512,
                 embed_dim: int = 384):
        super(TableEmptyTransformerClassifier, self).__init__()
        
        self.num_points = num_points
        self.embed_dim = embed_dim
        
        # Point transformer for teapot
        self.teapot_transformer = Point3DTransformer(embed_dim=embed_dim)
        
        # Point transformer for table
        self.table_transformer = Point3DTransformer(embed_dim=embed_dim)
        
        # MLP classifier on combined features
        combined_dim = embed_dim * 2  # Features from both teapot and table
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1)  # Binary classification
        )
    
    def forward(self, teapot_points: torch.Tensor, table_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            teapot_points: Teapot point cloud (batch_size, num_points, 3)
            table_points: Table point cloud (batch_size, num_points, 3)
            
        Returns:
            Classification logits (batch_size, 1)
        """
        # Extract features from both point clouds
        teapot_features = self.teapot_transformer(teapot_points)  # (batch_size, embed_dim)
        table_features = self.table_transformer(table_points)    # (batch_size, embed_dim)
        
        # Combine features
        combined_features = torch.cat([teapot_features, table_features], dim=1)
        
        # Classify
        logits = self.classifier(combined_features)
        
        return logits


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


def predict_table_empty_transformer(teapot_pcd_path: str, 
                                   table_pcd_path: str, 
                                   model_path: str = None,
                                   device: str = "auto") -> Tuple[bool, float]:
    """Predict whether table is empty using transformer model.
    
    Args:
        teapot_pcd_path: Path to teapot point cloud (.ply)
        table_pcd_path: Path to table point cloud (.ply)  
        model_path: Path to trained model
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
    
    # Convert to tensors
    teapot_tensor = torch.FloatTensor(teapot_points).unsqueeze(0).to(device)  # (1, 512, 3)
    table_tensor = torch.FloatTensor(table_points).unsqueeze(0).to(device)    # (1, 512, 3)
    
    # Load model
    model = TableEmptyTransformerClassifier(num_points=512, embed_dim=384)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
    else:
        print("Warning: No trained model provided. Using randomly initialized model.")
        print("Train a model first using: python scripts/train_table_empty_transformer.py")
    
    model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        logits = model(teapot_tensor, table_tensor)
        probability = torch.sigmoid(logits).item()
        prediction = probability > 0.5
    
    return prediction, probability


def main():
    parser = argparse.ArgumentParser(description='Predict TableEmpty using Transformer')
    parser.add_argument('teapot_pcd', help='Path to teapot point cloud (.ply)')
    parser.add_argument('table_pcd', help='Path to table point cloud (.ply)')
    parser.add_argument('--model', help='Path to trained transformer model (.pth)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for inference')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    try:
        # Make prediction
        is_empty, confidence = predict_table_empty_transformer(
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