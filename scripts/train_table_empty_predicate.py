"""Train a neural classifier for TableEmpty(?teapot, ?table) predicate using real-world point cloud data.

This script trains a PointNet-based binary classifier to predict whether a table is empty
based on teapot and table point clouds from Real-World-Data.
"""

import argparse
import os
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import open3d as o3d
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import IVNTR modules
from predicators.settings import CFG
from predicators.gnn.pointnet_utils import PointNetEncoder
from predicators.ml_models import PyTorchBinaryClassifier


class TableEmptyPointCloudDataset(Dataset):
    """Dataset for TableEmpty predicate training using point clouds."""
    
    def __init__(self, data_dir: str, num_points: int = 1024):
        """
        Args:
            data_dir: Path to Real-World-Data directory
            num_points: Number of points to sample from each point cloud
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.samples = []
        
        # Load data samples
        self._load_samples()
        
    def _load_samples(self):
        """Load point cloud samples and create positive/negative examples."""
        
        # Load teapot point clouds
        teapot_dir = os.path.join(self.data_dir, "Teapot")
        teapot_clouds = self._load_point_clouds_from_dir(teapot_dir)
        
        # Load table empty point clouds (positive examples)
        table_empty_dir = os.path.join(self.data_dir, "TableEmpty")
        table_empty_clouds = self._load_point_clouds_from_dir(table_empty_dir)
        
        # Load table full point clouds (negative examples)  
        table_full_dir = os.path.join(self.data_dir, "TableFull")
        table_full_clouds = self._load_point_clouds_from_dir(table_full_dir)
        
        # Create positive examples (table is empty)
        min_samples = min(len(teapot_clouds), len(table_empty_clouds))
        for i in range(min_samples):
            # Combine teapot and empty table point clouds
            combined_pcd = np.vstack([teapot_clouds[i], table_empty_clouds[i]])
            self.samples.append((combined_pcd, 1))  # Label 1 = table empty
            
        # Create negative examples (table is full)
        min_samples = min(len(teapot_clouds), len(table_full_clouds))
        for i in range(min_samples):
            # Combine teapot and full table point clouds
            combined_pcd = np.vstack([teapot_clouds[i], table_full_clouds[i]])
            self.samples.append((combined_pcd, 0))  # Label 0 = table not empty
            
        logging.info(f"Created {len(self.samples)} samples: "
                    f"{sum(1 for _, label in self.samples if label == 1)} positive, "
                    f"{sum(1 for _, label in self.samples if label == 0)} negative")
    
    def _load_point_clouds_from_dir(self, directory: str) -> List[np.ndarray]:
        """Load all point clouds from a directory."""
        point_clouds = []
        
        if not os.path.exists(directory):
            logging.error(f"Directory not found: {directory}")
            return point_clouds
            
        # Get all .ply files
        ply_files = [f for f in os.listdir(directory) if f.endswith('.ply')]
        ply_files.sort()  # Ensure consistent ordering
        
        for ply_file in ply_files:
            file_path = os.path.join(directory, ply_file)
            try:
                # Load point cloud using Open3D
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points, dtype=np.float32)
                
                if len(points) == 0:
                    logging.warning(f"Empty point cloud: {file_path}")
                    continue
                    
                # Sample to target number of points
                points = self._sample_points(points, self.num_points // 2)  # Half for each object
                point_clouds.append(points)
                
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
                
        logging.info(f"Loaded {len(point_clouds)} point clouds from {directory}")
        return point_clouds
    
    def _sample_points(self, points: np.ndarray, target_points: int) -> np.ndarray:
        """Sample target number of points from point cloud."""
        if len(points) >= target_points:
            # Random sampling
            indices = np.random.choice(len(points), target_points, replace=False)
            return points[indices]
        else:
            # Repeat points to reach target
            repeat_times = target_points // len(points) + 1
            repeated = np.tile(points, (repeat_times, 1))
            return repeated[:target_points]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        point_cloud, label = self.samples[idx]
        
        # Convert to tensor and ensure correct shape
        point_cloud = torch.FloatTensor(point_cloud)  # Shape: (N, 3)
        
        # Transpose for PointNet (expects 3 x N)
        point_cloud = point_cloud.transpose(0, 1)  # Shape: (3, N)
        
        label = torch.LongTensor([label])
        
        return point_cloud, label


class TableEmptyPointNetClassifier(nn.Module):
    """PointNet-based classifier for TableEmpty predicate.
    
    Based on the PointNetMLPClassifier from predicators/gnn/neupi.py
    """
    
    def __init__(self, num_points: int = 1024, pointnet_layers: List[int] = None):
        super(TableEmptyPointNetClassifier, self).__init__()
        
        if pointnet_layers is None:
            pointnet_layers = [64, 128, 256]  # Default architecture: conv layers
            
        self.num_points = num_points
        self.pointnet_layers = pointnet_layers
        
        # PointNet encoder (following IVNTR pattern)
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
        
        # Store dimensions for debugging
        self.pointnet_output_dim = pointnet_output_dim
        self.mlp_layers = mlp_layers
        
        # Initialize weights (following IVNTR pattern)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -1.0, 1.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Point cloud tensor of shape (batch_size, 3, num_points)
            
        Returns:
            Binary classification logits of shape (batch_size, 1)
        """
        # PointNet feature extraction
        features = self.pointnet(x)  # Output: (batch_size, pointnet_output_dim)
        
        # Debug print (remove after fixing)
        # print(f"PointNet output shape: {features.shape}")
        # print(f"Expected input to fc1: {self.pointnet_output_dim}")
        
        # MLP classification head
        x = torch.relu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # Output: (batch_size, 1)
        
        return x


def train_table_empty_classifier(data_dir: str, 
                                output_path: str,
                                num_epochs: int = 100,
                                batch_size: int = 32,
                                learning_rate: float = 0.001,
                                num_points: int = 1024,
                                train_split: float = 0.8) -> Dict[str, Any]:
    """Train the TableEmpty predicate classifier.
    
    Args:
        data_dir: Path to Real-World-Data directory
        output_path: Path to save trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_points: Number of points per point cloud
        train_split: Fraction of data for training
        
    Returns:
        Dictionary with training results and metrics
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create dataset
    dataset = TableEmptyPointCloudDataset(data_dir, num_points)
    
    if len(dataset) == 0:
        raise ValueError("No data loaded! Check your data directory.")
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = TableEmptyPointNetClassifier(num_points=num_points)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    logging.info(f"Starting training with {len(train_dataset)} train samples, "
                f"{len(val_dataset)} validation samples")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predictions = torch.sigmoid(output) > 0.5
                
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_predictions)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'num_points': num_points
            }, output_path)
        
        # Logging
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}/{num_epochs}: "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Acc: {val_acc:.4f}, "
                        f"Best Val Acc: {best_val_acc:.4f}")
    
    # Final evaluation
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_targets, val_predictions, average='binary'
    )
    
    results = {
        'best_val_accuracy': best_val_acc,
        'final_train_loss': avg_train_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'num_epochs': num_epochs,
        'num_samples': len(dataset)
    }
    
    logging.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    logging.info(f"Final metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return results


def load_trained_classifier(model_path: str, num_points: int = 1024) -> TableEmptyPointNetClassifier:
    """Load a trained TableEmpty classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TableEmptyPointNetClassifier(num_points=num_points)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logging.info(f"Loaded trained model from {model_path}")
    logging.info(f"Model achieved {checkpoint['val_acc']:.4f} validation accuracy")
    
    return model


def test_classifier_on_sample(model: TableEmptyPointNetClassifier, 
                             teapot_pcd_path: str, 
                             table_pcd_path: str) -> Tuple[bool, float]:
    """Test the classifier on a single sample."""
    device = next(model.parameters()).device
    
    # Load point clouds
    teapot_pcd = o3d.io.read_point_cloud(teapot_pcd_path)
    table_pcd = o3d.io.read_point_cloud(table_pcd_path)
    
    teapot_points = np.asarray(teapot_pcd.points, dtype=np.float32)
    table_points = np.asarray(table_pcd.points, dtype=np.float32)
    
    # Combine and sample points
    combined_points = np.vstack([teapot_points, table_points])
    if len(combined_points) > model.num_points:
        indices = np.random.choice(len(combined_points), model.num_points, replace=False)
        combined_points = combined_points[indices]
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(combined_points).transpose(0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
        prediction = probability > 0.5
    
    return prediction, probability


def main():
    parser = argparse.ArgumentParser(description='Train TableEmpty predicate classifier')
    parser.add_argument('--data_dir', default='Real-World-Data', 
                       help='Path to Real-World-Data directory')
    parser.add_argument('--output_path', default='table_empty_classifier.pth',
                       help='Path to save trained model')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='Number of points per point cloud')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test a trained model')
    parser.add_argument('--model_path', 
                       help='Path to trained model for testing')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.test_only:
        if not args.model_path:
            raise ValueError("Must provide --model_path for testing")
        
        # Load and test model
        model = load_trained_classifier(args.model_path, args.num_points)
        
        # Test on a sample
        teapot_sample = os.path.join(args.data_dir, "Teapot", "cloud_1.ply")
        empty_table_sample = os.path.join(args.data_dir, "TableEmpty", "cloud_1.ply")
        full_table_sample = os.path.join(args.data_dir, "TableFull", "cloud_1.ply")
        
        if all(os.path.exists(path) for path in [teapot_sample, empty_table_sample, full_table_sample]):
            pred_empty, prob_empty = test_classifier_on_sample(model, teapot_sample, empty_table_sample)
            pred_full, prob_full = test_classifier_on_sample(model, teapot_sample, full_table_sample)
            
            print(f"Empty table prediction: {pred_empty} (probability: {prob_empty:.4f})")
            print(f"Full table prediction: {pred_full} (probability: {prob_full:.4f})")
        else:
            print("Sample files not found for testing")
    
    else:
        # Train model
        results = train_table_empty_classifier(
            data_dir=args.data_dir,
            output_path=args.output_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_points=args.num_points,
            train_split=args.train_split
        )
        
        print(f"\nTraining Results:")
        print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Model saved to: {args.output_path}")


if __name__ == "__main__":
    main()