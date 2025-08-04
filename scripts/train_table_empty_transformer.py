"""Train TableEmpty predicate using Point Cloud Transformer with pretrained weights + MLP.

This script uses a pretrained Point Cloud Transformer to extract features from 
point clouds, then trains a simple MLP classifier for the TableEmpty predicate.
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
import timm


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
            # Store separately for transformer processing
            self.samples.append((teapot_clouds[i], table_empty_clouds[i], 1))  # Label 1 = table empty
            
        # Create negative examples (table is full)
        min_samples = min(len(teapot_clouds), len(table_full_clouds))
        for i in range(min_samples):
            # Store separately for transformer processing
            self.samples.append((teapot_clouds[i], table_full_clouds[i], 0))  # Label 0 = table not empty
            
        logging.info(f"Created {len(self.samples)} samples: "
                    f"{sum(1 for _, _, label in self.samples if label == 1)} positive, "
                    f"{sum(1 for _, _, label in self.samples if label == 0)} negative")
    
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teapot_pcd, table_pcd, label = self.samples[idx]
        
        # Convert to tensors
        teapot_tensor = torch.FloatTensor(teapot_pcd)  # Shape: (N/2, 3)
        table_tensor = torch.FloatTensor(table_pcd)    # Shape: (N/2, 3)
        label_tensor = torch.LongTensor([label])
        
        return teapot_tensor, table_tensor, label_tensor


class Point3DTransformer(nn.Module):
    """Simple 3D Point Transformer for point cloud feature extraction.
    
    This is a simplified version that can be used with or without pretrained weights.
    """
    
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
                 embed_dim: int = 384,
                 pretrained_path: str = None,
                 freeze_transformer: bool = True):
        super(TableEmptyTransformerClassifier, self).__init__()
        
        self.num_points = num_points
        self.embed_dim = embed_dim
        
        # Point transformer for teapot
        self.teapot_transformer = Point3DTransformer(embed_dim=embed_dim)
        
        # Point transformer for table (can share weights or use separate)
        self.table_transformer = Point3DTransformer(embed_dim=embed_dim)
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
            logging.info(f"Loaded pretrained weights from {pretrained_path}")
        
        # Freeze transformer weights if specified
        if freeze_transformer:
            self._freeze_transformers()
            logging.info("Frozen transformer weights")
        
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
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained transformer weights."""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Try to load state dict (adjust based on your pretrained model format)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights for both transformers (they can share the same pretrained weights)
            self.teapot_transformer.load_state_dict(state_dict, strict=False)
            self.table_transformer.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            logging.warning(f"Could not load pretrained weights: {e}")
            logging.info("Continuing with random initialization")
    
    def _freeze_transformers(self):
        """Freeze transformer parameters."""
        for param in self.teapot_transformer.parameters():
            param.requires_grad = False
        for param in self.table_transformer.parameters():
            param.requires_grad = False
    
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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


def train_transformer_classifier(data_dir: str,
                                output_path: str,
                                pretrained_path: str = None,
                                num_epochs: int = 50,
                                batch_size: int = 16,
                                learning_rate: float = 0.001,
                                num_points: int = 512,
                                train_split: float = 0.8,
                                freeze_transformer: bool = True) -> Dict[str, Any]:
    """Train the TableEmpty transformer classifier."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create dataset
    dataset = TableEmptyPointCloudDataset(data_dir, num_points * 2)  # *2 because we split into teapot/table
    
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
    model = TableEmptyTransformerClassifier(
        num_points=num_points,
        embed_dim=384,
        pretrained_path=pretrained_path,
        freeze_transformer=freeze_transformer
    )
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # Only optimize classifier parameters if transformer is frozen
    if freeze_transformer:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
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
        
        for batch_idx, (teapot_data, table_data, target) in enumerate(train_loader):
            teapot_data = teapot_data.to(device)
            table_data = table_data.to(device)
            target = target.to(device).float()
            
            optimizer.zero_grad()
            output = model(teapot_data, table_data)
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
            for teapot_data, table_data, target in val_loader:
                teapot_data = teapot_data.to(device)
                table_data = table_data.to(device)
                target = target.to(device)
                
                output = model(teapot_data, table_data)
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
        if epoch % 5 == 0:
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


def main():
    parser = argparse.ArgumentParser(description='Train TableEmpty predicate with Transformer')
    parser.add_argument('--data_dir', default='Real-World-Data',
                       help='Path to Real-World-Data directory')
    parser.add_argument('--output_path', default='table_empty_transformer.pth',
                       help='Path to save trained model')
    parser.add_argument('--pretrained_path',
                       help='Path to pretrained transformer weights')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_points', type=int, default=512,
                       help='Number of points per point cloud')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--no_freeze', action='store_true',
                       help='Do not freeze transformer weights (fine-tune end-to-end)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Train model
    results = train_transformer_classifier(
        data_dir=args.data_dir,
        output_path=args.output_path,
        pretrained_path=args.pretrained_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_points=args.num_points,
        train_split=args.train_split,
        freeze_transformer=not args.no_freeze
    )
    
    print(f"\nTraining Results:")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Model saved to: {args.output_path}")


if __name__ == "__main__":
    main()