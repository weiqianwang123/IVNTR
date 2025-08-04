"""Train TableEmpty predicate using Point-BERT pretrained features + MLP.

This script uses a pretrained Point-BERT model to extract features from 
point clouds, then trains a simple MLP classifier for the TableEmpty predicate.
"""

import argparse
import os
import sys
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import open3d as o3d
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import yaml

# Add Point-BERT to path
POINT_BERT_PATH = os.path.join(os.path.dirname(__file__), '..', 'Point-BERT')
if os.path.exists(POINT_BERT_PATH):
    sys.path.insert(0, POINT_BERT_PATH)

# Import Point-BERT components from local codebase
try:
    from models.Point_BERT import Point_BERT
    from utils.config import merge_new_config
    from easydict import EasyDict
    POINTBERT_AVAILABLE = True
    print("Successfully imported Point-BERT from local codebase")
except ImportError as e:
    print(f"Point-BERT modules not found: {e}")
    print("Using simplified transformer instead.")
    POINTBERT_AVAILABLE = False


class TableEmptyPointCloudDataset(Dataset):
    """Dataset for TableEmpty predicate training using point clouds."""
    
    def __init__(self, data_dir: str, num_points: int = 8192):
        """
        Args:
            data_dir: Path to Real-World-Data directory
            num_points: Number of points for Point-BERT (typically 8192)
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.samples = []
        
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
            
        ply_files = [f for f in os.listdir(directory) if f.endswith('.ply')]
        ply_files.sort()
        
        for ply_file in ply_files:
            file_path = os.path.join(directory, ply_file)
            try:
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points, dtype=np.float32)
                
                if len(points) == 0:
                    logging.warning(f"Empty point cloud: {file_path}")
                    continue
                    
                # Sample points (half of target for combination later)
                points = self._sample_points(points, self.num_points // 2)
                point_clouds.append(points)
                
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
                
        logging.info(f"Loaded {len(point_clouds)} point clouds from {directory}")
        return point_clouds
    
    def _sample_points(self, points: np.ndarray, target_points: int) -> np.ndarray:
        """Sample target number of points from point cloud."""
        if len(points) >= target_points:
            indices = np.random.choice(len(points), target_points, replace=False)
            return points[indices]
        else:
            repeat_times = target_points // len(points) + 1
            repeated = np.tile(points, (repeat_times, 1))
            return repeated[:target_points]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        point_cloud, label = self.samples[idx]
        
        # Ensure we have exactly the right number of points
        if len(point_cloud) != self.num_points:
            point_cloud = self._sample_points(point_cloud, self.num_points)
        
        # Convert to tensor
        point_cloud = torch.FloatTensor(point_cloud)  # Shape: (num_points, 3)
        label = torch.LongTensor([label])
        
        return point_cloud, label


class PointBERTFeatureExtractor(nn.Module):
    """Feature extractor using pretrained Point-BERT."""
    
    def __init__(self, config_path=None, pretrained_path=None):
        super().__init__()
        
        self.config = None
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        
        if POINTBERT_AVAILABLE and self.config:
            # Use actual Point-BERT with loaded config
            try:
                # Point-BERT expects the full model config, not just transformer_config
                model_config = EasyDict(self.config.model)
                self.backbone = Point_BERT(model_config)
                logging.info("Created Point-BERT model with loaded config")
            except Exception as e:
                logging.warning(f"Failed to create Point-BERT with config: {e}")
                logging.warning(f"Error details: {str(e)}")
                self.backbone = self._create_simple_transformer()
        else:
            # Fallback: simplified transformer
            self.backbone = self._create_simple_transformer()
        
        # Load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.feature_dim = self._get_feature_dim()
    
    def _load_config(self, config_path):
        """Load Point-BERT config from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert to EasyDict format expected by Point-BERT
            if POINTBERT_AVAILABLE:
                config = EasyDict(config_dict)
                logging.info(f"Loaded Point-BERT config from {config_path}")
                return config
            else:
                logging.info(f"Loaded config from {config_path} (fallback mode)")
                return config_dict
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}")
            return None
    
    def _create_simple_transformer(self):
        """Create a simplified transformer if Point-BERT is not available."""
        class SimplePointTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(3, 384)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(384, 8, 1536, batch_first=True),
                    num_layers=6
                )
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.feature_dim = 384
            
            def forward(self, pts):
                # pts: (B, N, 3)
                x = self.embed(pts)  # (B, N, 384)
                x = self.transformer(x)  # (B, N, 384)
                x = x.transpose(1, 2)  # (B, 384, N)
                x = self.global_pool(x).squeeze(-1)  # (B, 384)
                return x
        
        return SimplePointTransformer()
    
    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained Point-BERT weights."""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'base_model' in checkpoint:
                state_dict = checkpoint['base_model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Clean up state dict keys
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # Remove 'module.' prefix
                clean_state_dict[k] = v
            
            # Check if this is a Point-BERT model vs our fallback
            model_keys = set(self.backbone.state_dict().keys())
            checkpoint_keys = set(clean_state_dict.keys())
            
            # If very few keys match, this is likely Point-BERT weights vs fallback model
            matching_keys = model_keys.intersection(checkpoint_keys)
            if len(matching_keys) < len(model_keys) * 0.5:  # Less than 50% keys match
                logging.warning(f"Architecture mismatch detected:")
                logging.warning(f"  Model has {len(model_keys)} parameters")
                logging.warning(f"  Checkpoint has {len(checkpoint_keys)} parameters") 
                logging.warning(f"  Only {len(matching_keys)} keys match")
                logging.warning("This usually means Point-BERT module is not available but Point-BERT weights were provided")
                logging.info("Skipping weight loading to avoid errors. Install Point-BERT module for full compatibility.")
                return
            
            # Load weights (strict=False to handle missing keys)
            missing_keys = self.backbone.load_state_dict(clean_state_dict, strict=False)
            
            if hasattr(missing_keys, 'missing_keys') and missing_keys.missing_keys:
                if len(missing_keys.missing_keys) > 10:  # Only show first few if many missing
                    logging.warning(f"Missing {len(missing_keys.missing_keys)} keys when loading Point-BERT")
                    logging.debug(f"First few missing keys: {missing_keys.missing_keys[:5]}")
                else:
                    logging.warning(f"Missing keys when loading Point-BERT: {missing_keys.missing_keys}")
            
            logging.info(f"Successfully loaded pretrained weights from {pretrained_path}")
            
        except Exception as e:
            logging.warning(f"Failed to load pretrained weights: {e}")
            logging.info("Continuing with random initialization")
    
    def _get_feature_dim(self):
        """Get the feature dimension from the backbone by testing actual output."""
        # Test with a dummy input to get actual output dimension
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 8192, 3)  # Batch=1, Points=8192, Coords=3
                if hasattr(self.backbone, 'forward_eval'):
                    output = self.backbone.forward_eval(dummy_input)
                else:
                    output = self.backbone(dummy_input)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                actual_dim = output.shape[-1]
                logging.info(f"Detected actual Point-BERT output dimension: {actual_dim}")
                return actual_dim
        except Exception as e:
            logging.warning(f"Failed to detect feature dimension by testing: {e}")
        
        # Fallback: try to get from config
        if self.config:
            transformer_config = self.config.get('model', {}).get('transformer_config', {})
            if 'trans_dim' in transformer_config:
                return transformer_config['trans_dim']
            elif 'cls_dim' in transformer_config:
                return transformer_config['cls_dim']
        
        # Try to get from backbone attributes
        if hasattr(self.backbone, 'feature_dim'):
            return self.backbone.feature_dim
        elif hasattr(self.backbone, 'trans_dim'):
            return self.backbone.trans_dim
        elif hasattr(self.backbone, 'cls_dim'):
            return self.backbone.cls_dim
        else:
            # Default fallback
            return 384
    
    def forward(self, pts):
        """Extract features using Point-BERT."""
        with torch.no_grad():
            if POINTBERT_AVAILABLE and hasattr(self.backbone, 'forward'):
                # Point-BERT forward pass
                try:
                    # Point-BERT expects input shape (B, N, 3)
                    if pts.dim() == 3 and pts.shape[-1] == 3:
                        # Input is already (B, N, 3)
                        point_input = pts
                    else:
                        # Need to reshape or handle differently
                        point_input = pts
                    
                    # Point-BERT typically returns a dictionary or features
                    output = self.backbone(point_input)
                    
                    # Handle different return formats from Point-BERT
                    if isinstance(output, dict):
                        # Point-BERT might return dict with 'features' or 'cls_tokens'
                        if 'features' in output:
                            features = output['features']
                        elif 'cls_tokens' in output:
                            features = output['cls_tokens']
                        else:
                            # Take the first value in the dict
                            features = list(output.values())[0]
                    elif isinstance(output, tuple):
                        features = output[0]  # Take first element if tuple
                    else:
                        features = output
                        
                    # Ensure we have the right shape - global features should be (B, D)
                    if features.dim() > 2:
                        features = features.mean(dim=1)  # Global average pooling if needed
                        
                    return features
                except Exception as e:
                    logging.warning(f"Point-BERT forward failed: {e}")
                    # Fallback to regular forward
                    return self.backbone(pts)
            else:
                # Fallback transformer forward
                features = self.backbone(pts)
                if isinstance(features, tuple):
                    features = features[0]
                return features


class TableEmptyMLPClassifier(nn.Module):
    """MLP classifier for TableEmpty predicate using Point-BERT features."""
    
    def __init__(self, 
                 feature_dim: int = 384,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()
        
        # Build MLP layers
        layers = []
        input_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize MLP weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features):
        """Forward pass through MLP."""
        return self.classifier(features)


class TableEmptyPointBERTModel(nn.Module):
    """Complete model: Point-BERT feature extractor + MLP classifier."""
    
    def __init__(self, 
                 config_path=None,
                 pretrained_path=None,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = PointBERTFeatureExtractor(
            config_path=config_path,
            pretrained_path=pretrained_path
        )
        
        # MLP classifier
        self.classifier = TableEmptyMLPClassifier(
            feature_dim=self.feature_extractor.feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    
    def forward(self, pts):
        """Forward pass: extract features then classify."""
        features = self.feature_extractor(pts)
        logits = self.classifier(features)
        return logits


def train_pointbert_classifier(data_dir: str,
                              output_path: str,
                              config_path: str = None,
                              pretrained_path: str = None,
                              num_epochs: int = 50,
                              batch_size: int = 8,  # Smaller batch size for Point-BERT
                              learning_rate: float = 0.001,
                              num_points: int = 8192,  # Point-BERT typically uses 8192
                              train_split: float = 0.8) -> Dict[str, Any]:
    """Train the TableEmpty Point-BERT classifier."""
    
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
    model = TableEmptyPointBERTModel(
        config_path=config_path,
        pretrained_path=pretrained_path,
        hidden_dims=[512, 256, 128],
        dropout=0.3
    )
    model.to(device)
    
    # Loss function and optimizer (only optimize classifier parameters)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    logging.info(f"Starting training with {len(train_dataset)} train samples, "
                f"{len(val_dataset)} validation samples")
    logging.info(f"Point-BERT feature dim: {model.feature_extractor.feature_dim}")
    
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
                'feature_extractor_state_dict': model.feature_extractor.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'num_points': num_points,
                'feature_dim': model.feature_extractor.feature_dim
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
        'num_samples': len(dataset),
        'feature_dim': model.feature_extractor.feature_dim
    }
    
    logging.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    logging.info(f"Final metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train TableEmpty predicate with Point-BERT')
    parser.add_argument('--data_dir', default='Real-World-Data',
                       help='Path to Real-World-Data directory')
    parser.add_argument('--output_path', default='table_empty_pointbert.pth',
                       help='Path to save trained model')
    parser.add_argument('--config_path', default='scripts/bert_config.yaml',
                       help='Path to Point-BERT config file')
    parser.add_argument('--pretrained_path', default='scripts/weights/Point-BERT.pth',
                       help='Path to pretrained Point-BERT weights')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size (smaller for Point-BERT)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_points', type=int, default=8192,
                       help='Number of points for Point-BERT (typically 8192)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check if files exist
    if not os.path.exists(args.config_path):
        logging.warning(f"Config file not found: {args.config_path}")
        logging.info("Continuing without config (will use fallback)")
        args.config_path = None
    
    if not os.path.exists(args.pretrained_path):
        logging.warning(f"Pretrained Point-BERT model not found: {args.pretrained_path}")
        logging.info("Continuing with random initialization")
        args.pretrained_path = None
    
    # Train model
    results = train_pointbert_classifier(
        data_dir=args.data_dir,
        output_path=args.output_path,
        config_path=args.config_path,
        pretrained_path=args.pretrained_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_points=args.num_points,
        train_split=args.train_split
    )
    
    print(f"\nPoint-BERT Training Results:")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Feature Dimension: {results['feature_dim']}")
    print(f"Model saved to: {args.output_path}")


if __name__ == "__main__":
    main()