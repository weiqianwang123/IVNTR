"""Integrate the trained TableEmpty classifier into IVNTR as a neural predicate.

This script shows how to use the trained TableEmpty classifier within the IVNTR framework
as a proper Predicate that can be used for planning and learning.
"""

import os
import logging
from typing import List, Sequence
import numpy as np
import torch
import open3d as o3d

from predicators.structs import Predicate, Object, State, Type
from predicators.ml_models import LearnedPredicateClassifier
from scripts.train_table_empty_predicate import TableEmptyPointNetClassifier, load_trained_classifier


class TableEmptyNeuralPredicate:
    """Neural predicate wrapper for TableEmpty(?teapot, ?table)."""
    
    def __init__(self, model_path: str, num_points: int = 1024):
        """
        Initialize the neural predicate.
        
        Args:
            model_path: Path to trained TableEmpty classifier
            num_points: Number of points expected by the model
        """
        self.model = load_trained_classifier(model_path, num_points)
        self.device = next(self.model.parameters()).device
        self.num_points = num_points
        
        logging.info(f"Initialized TableEmpty neural predicate with model from {model_path}")
    
    def __call__(self, state: State, objects: Sequence[Object]) -> bool:
        """
        Predicate classifier function compatible with IVNTR.
        
        Args:
            state: Current state containing object point clouds
            objects: [teapot_object, table_object]
            
        Returns:
            True if table is empty, False otherwise
        """
        if len(objects) != 2:
            raise ValueError(f"TableEmpty predicate expects 2 objects, got {len(objects)}")
        
        teapot_obj, table_obj = objects
        
        # Extract point clouds from state
        teapot_pcd = self._extract_point_cloud_from_state(state, teapot_obj)
        table_pcd = self._extract_point_cloud_from_state(state, table_obj)
        
        if teapot_pcd is None or table_pcd is None:
            logging.warning("Could not extract point clouds from state")
            return False
        
        # Combine point clouds
        combined_pcd = np.vstack([teapot_pcd, table_pcd])
        
        # Sample to target number of points
        if len(combined_pcd) > self.num_points:
            indices = np.random.choice(len(combined_pcd), self.num_points, replace=False)
            combined_pcd = combined_pcd[indices]
        elif len(combined_pcd) < self.num_points:
            # Repeat points to reach target
            repeat_times = self.num_points // len(combined_pcd) + 1
            combined_pcd = np.tile(combined_pcd, (repeat_times, 1))[:self.num_points]
        
        # Convert to tensor and predict
        input_tensor = torch.FloatTensor(combined_pcd).transpose(0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = probability > 0.5
        
        return prediction
    
    def _extract_point_cloud_from_state(self, state: State, obj: Object) -> np.ndarray:
        """Extract point cloud data from state for given object."""
        
        # Method 1: Direct point cloud feature (following tools_pcd pattern)
        if 'pcd' in obj.type.feature_names:
            pcd_idx = obj.type.feature_names.index('pcd')
            pcd_data = state[obj][pcd_idx]
            if isinstance(pcd_data, np.ndarray) and pcd_data.shape[1] == 3:
                return pcd_data
        
        # Method 2: Individual point coordinates (pcd_0, pcd_1, pcd_2, ...)
        pcd_features = [name for name in obj.type.feature_names if name.startswith('pcd_')]
        if pcd_features:
            # Reconstruct point cloud from individual coordinates
            num_coords = len(pcd_features)
            num_points = num_coords // 3
            
            points = []
            for i in range(num_points):
                x_idx = obj.type.feature_names.index(f'pcd_{i*3}')
                y_idx = obj.type.feature_names.index(f'pcd_{i*3+1}')
                z_idx = obj.type.feature_names.index(f'pcd_{i*3+2}')
                
                x = state[obj][x_idx]
                y = state[obj][y_idx] 
                z = state[obj][z_idx]
                
                points.append([x, y, z])
            
            return np.array(points, dtype=np.float32)
        
        # Method 3: Load from file path (if stored as path in state)
        if hasattr(obj, 'point_cloud_path'):
            pcd = o3d.io.read_point_cloud(obj.point_cloud_path)
            return np.asarray(pcd.points, dtype=np.float32)
        
        logging.warning(f"Could not extract point cloud for object {obj.name}")
        return None


def create_table_empty_predicate(model_path: str, 
                                teapot_type: Type, 
                                table_type: Type) -> Predicate:
    """Create a TableEmpty predicate using the trained neural classifier.
    
    Args:
        model_path: Path to trained TableEmpty model
        teapot_type: Type definition for teapot objects
        table_type: Type definition for table objects
        
    Returns:
        Predicate object that can be used in IVNTR
    """
    
    # Create neural predicate classifier
    neural_classifier = TableEmptyNeuralPredicate(model_path)
    
    # Create IVNTR predicate
    predicate = Predicate(
        name="TableEmpty",
        types=[teapot_type, table_type],
        _classifier=neural_classifier
    )
    
    return predicate


def example_usage():
    """Example of how to use the TableEmpty neural predicate."""
    
    # Define object types with point cloud features
    teapot_type = Type("teapot", [
        "pose_x", "pose_y", "pose_z",
        "pcd",  # Full point cloud stored as (N, 3) array
    ])
    
    table_type = Type("table", [
        "pose_x", "pose_y", "pose_z", 
        "pcd",  # Full point cloud stored as (N, 3) array
    ])
    
    # Create objects
    teapot = Object("teapot1", teapot_type)
    table = Object("table1", table_type)
    
    # Create predicate (assuming model is trained)
    model_path = "table_empty_classifier.pth"
    if os.path.exists(model_path):
        table_empty_pred = create_table_empty_predicate(model_path, teapot_type, table_type)
        
        # Example state with point cloud data
        # In practice, this would come from your data loading pipeline
        teapot_pcd = np.random.random((512, 3)).astype(np.float32)
        table_pcd = np.random.random((512, 3)).astype(np.float32)
        
        state_data = {
            teapot: np.array([1.0, 0.5, 0.3] + [teapot_pcd]),  # pose + pcd
            table: np.array([0.0, 0.0, 0.0] + [table_pcd])     # pose + pcd
        }
        
        from predicators.structs import State
        state = State(state_data)
        
        # Test predicate
        result = table_empty_pred.holds(state, [teapot, table])
        print(f"TableEmpty({teapot.name}, {table.name}) = {result}")
        
    else:
        print(f"Model not found at {model_path}. Train the model first using:")
        print("python scripts/train_table_empty_predicate.py")


def integrate_with_environment():
    """Example of integrating TableEmpty predicate with an IVNTR environment."""
    
    from predicators.envs import BaseEnv
    from predicators.structs import GroundAtom
    
    class TableTeapotEnv(BaseEnv):
        """Example environment with TableEmpty predicate."""
        
        def __init__(self, table_empty_model_path: str):
            super().__init__()
            
            # Define types
            self.teapot_type = Type("teapot", ["pose_x", "pose_y", "pose_z", "pcd"])
            self.table_type = Type("table", ["pose_x", "pose_y", "pose_z", "pcd"])
            
            # Create neural predicate
            self.table_empty_pred = create_table_empty_predicate(
                table_empty_model_path, self.teapot_type, self.table_type
            )
        
        @classmethod
        def get_name(cls) -> str:
            return "table_teapot"
        
        @property
        def predicates(self):
            return {self.table_empty_pred}
        
        @property
        def goal_predicates(self):
            return {self.table_empty_pred}
        
        @property
        def types(self):
            return {self.teapot_type, self.table_type}
        
        # ... implement other required methods
    
    # Usage
    if os.path.exists("table_empty_classifier.pth"):
        env = TableTeapotEnv("table_empty_classifier.pth")
        print(f"Created environment with TableEmpty predicate: {env.table_empty_pred}")


def main():
    """Main function demonstrating predicate integration."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("TableEmpty Neural Predicate Integration")
    print("=" * 50)
    
    print("\n1. Example usage:")
    example_usage()
    
    print("\n2. Environment integration:")
    integrate_with_environment()
    
    print("\nTo train the classifier first, run:")
    print("python scripts/train_table_empty_predicate.py --data_dir Real-World-Data")
    print("\nTo test the trained classifier, run:")
    print("python scripts/train_table_empty_predicate.py --test_only --model_path table_empty_classifier.pth")


if __name__ == "__main__":
    main()