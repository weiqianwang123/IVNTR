"""Direct integration of real-world demonstration data with state-action-state format.

This assumes you already have demonstrations structured as:
state0 -> action0 -> state1 -> action1 -> state2 -> ...
where each state contains 1 point cloud and 1 image.
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Set, Any, Union
import numpy as np
from PIL import Image
import open3d as o3d

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Dataset, LowLevelTrajectory, Object, \
    ParameterizedOption, State, Task


def create_direct_real_world_demo_data(env: BaseEnv, train_tasks: List[Task],
                                      known_options: Set[ParameterizedOption]) -> Dataset:
    """Create dataset directly from real-world state-action-state demonstrations."""
    trajectories = []
    
    demo_data_dir = CFG.real_world_demo_dir
    
    for task_idx, task in enumerate(train_tasks):
        if task_idx >= CFG.max_initial_demos:
            break
            
        # Try multiple possible file formats
        demo_file = None
        for ext in ['.pkl', '.json', '.npz']:
            candidate = os.path.join(demo_data_dir, f"demo_{task_idx}{ext}")
            if os.path.exists(candidate):
                demo_file = candidate
                break
        
        if demo_file:
            trajectory = _load_direct_trajectory(demo_file, env, task_idx)
            if trajectory is not None:
                trajectories.append(trajectory)
                logging.info(f"Loaded direct real-world demo {task_idx}")
        else:
            logging.warning(f"No real-world demo found for task {task_idx}")
    
    logging.info(f"Created {len(trajectories)} direct real-world demonstrations")
    return Dataset(trajectories)


def _load_direct_trajectory(demo_file: str, env: BaseEnv, task_idx: int) -> LowLevelTrajectory:
    """Load trajectory directly from your existing demonstration format."""
    
    # Determine file format and load
    if demo_file.endswith('.pkl'):
        with open(demo_file, 'rb') as f:
            demo_data = pickle.load(f)
    elif demo_file.endswith('.json'):
        with open(demo_file, 'r') as f:
            demo_data = json.load(f)
    elif demo_file.endswith('.npz'):
        demo_data = np.load(demo_file, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file format: {demo_file}")
    
    # Convert your demonstration format to IVNTR format
    states = []
    actions = []
    
    # Assuming your data structure has 'states' and 'actions' keys
    # Adjust these based on your actual data format
    raw_states = demo_data.get('states', demo_data.get('state_sequence', []))
    raw_actions = demo_data.get('actions', demo_data.get('action_sequence', []))
    
    # Process each state
    for i, raw_state in enumerate(raw_states):
        state = _convert_raw_state_to_ivntr_state(raw_state, env)
        states.append(state)
    
    # Process each action
    for raw_action in raw_actions:
        action = _convert_raw_action_to_ivntr_action(raw_action, env)
        actions.append(action)
    
    # Ensure invariant: len(states) == len(actions) + 1
    if len(states) != len(actions) + 1:
        logging.warning(f"State-action length mismatch in demo {task_idx}: "
                       f"{len(states)} states, {len(actions)} actions")
        # Adjust by truncating the longer sequence
        min_len = min(len(states), len(actions) + 1)
        states = states[:min_len]
        actions = actions[:min_len-1]
    
    return LowLevelTrajectory(
        _states=states,
        _actions=actions,
        _is_demo=True,
        _train_task_idx=task_idx
    )


def _convert_raw_state_to_ivntr_state(raw_state: Dict[str, Any], env: BaseEnv) -> State:
    """Convert your raw state format to IVNTR State format.
    
    Assumes raw_state contains:
    - point_cloud: numpy array or file path
    - image: PIL Image, numpy array, or file path  
    - object_poses: dictionary of object positions
    - other_features: any other state information
    """
    
    # Get environment objects
    objects = env.get_objects()
    state_dict = {}
    
    # Process point cloud data
    pcd_data = _process_point_cloud(raw_state.get('point_cloud', raw_state.get('pcd')))
    
    # Process image data
    image_data = _process_image(raw_state.get('image', raw_state.get('rgb_image')))
    
    # Build state dictionary for each object
    for obj in objects:
        obj_features = {}
        
        # Add traditional pose/physical features
        if 'object_poses' in raw_state and obj.name in raw_state['object_poses']:
            pose_data = raw_state['object_poses'][obj.name]
            for feat_name in obj.type.feature_names:
                if feat_name in ['pose_x', 'pose_y', 'pose_z']:
                    obj_features[feat_name] = pose_data.get(feat_name, 0.0)
                elif feat_name in ['color_r', 'color_g', 'color_b']:
                    obj_features[feat_name] = pose_data.get(feat_name, 0.0)
                elif feat_name in ['fingers', 'goal_achieved']:
                    obj_features[feat_name] = pose_data.get(feat_name, 0.0)
        
        # Add multimodal features
        for feat_name in obj.type.feature_names:
            if feat_name == 'pcd':
                # Store full point cloud
                obj_features[feat_name] = pcd_data if pcd_data is not None else np.zeros((1024, 3))
            elif feat_name.startswith('pcd_'):
                # Store individual point coordinates (following tools_pcd pattern)
                coord_idx = int(feat_name.split('_')[1])
                point_idx = coord_idx // 3
                coord_type = coord_idx % 3
                if pcd_data is not None and point_idx < len(pcd_data):
                    obj_features[feat_name] = float(pcd_data[point_idx, coord_type])
                else:
                    obj_features[feat_name] = 0.0
            elif feat_name.startswith('image_') or feat_name.startswith('rgb_'):
                # Store image features
                if image_data is not None:
                    img_idx = int(feat_name.split('_')[1]) if '_' in feat_name else 0
                    if img_idx < len(image_data):
                        obj_features[feat_name] = float(image_data[img_idx])
                    else:
                        obj_features[feat_name] = 0.0
                else:
                    obj_features[feat_name] = 0.0
            elif feat_name not in obj_features:
                # Default value for unspecified features
                obj_features[feat_name] = 0.0
        
        state_dict[obj] = obj_features
    
    return utils.create_state_from_dict(state_dict)


def _process_point_cloud(pcd_input: Union[str, np.ndarray, None]) -> np.ndarray:
    """Process point cloud data from various input formats."""
    if pcd_input is None:
        return np.zeros((CFG.pcd_dim, 3), dtype=np.float32)
    
    if isinstance(pcd_input, str):
        # File path
        if os.path.exists(pcd_input):
            if pcd_input.endswith('.ply') or pcd_input.endswith('.pcd'):
                pcd = o3d.io.read_point_cloud(pcd_input)
                points = np.asarray(pcd.points, dtype=np.float32)
            elif pcd_input.endswith('.npy'):
                points = np.load(pcd_input).astype(np.float32)
            else:
                logging.warning(f"Unknown point cloud format: {pcd_input}")
                return np.zeros((CFG.pcd_dim, 3), dtype=np.float32)
        else:
            logging.warning(f"Point cloud file not found: {pcd_input}")
            return np.zeros((CFG.pcd_dim, 3), dtype=np.float32)
    elif isinstance(pcd_input, np.ndarray):
        # Direct numpy array
        points = pcd_input.astype(np.float32)
    else:
        logging.warning(f"Unknown point cloud type: {type(pcd_input)}")
        return np.zeros((CFG.pcd_dim, 3), dtype=np.float32)
    
    # Ensure correct shape and size
    if len(points.shape) != 2 or points.shape[1] != 3:
        logging.warning(f"Invalid point cloud shape: {points.shape}, expected (N, 3)")
        return np.zeros((CFG.pcd_dim, 3), dtype=np.float32)
    
    # Resample to target size
    target_points = CFG.pcd_dim
    if len(points) > target_points:
        # Downsample
        indices = np.linspace(0, len(points)-1, target_points, dtype=int)
        points = points[indices]
    elif len(points) < target_points:
        # Upsample by repeating points
        if len(points) > 0:
            repeat_times = target_points // len(points) + 1
            points = np.tile(points, (repeat_times, 1))[:target_points]
        else:
            points = np.zeros((target_points, 3), dtype=np.float32)
    
    return points


def _process_image(image_input: Union[str, np.ndarray, Image.Image, None]) -> np.ndarray:
    """Process image data from various input formats."""
    if image_input is None:
        return np.zeros(CFG.image_feature_dim, dtype=np.float32)
    
    if isinstance(image_input, str):
        # File path
        if os.path.exists(image_input):
            img = Image.open(image_input)
            img_array = np.array(img, dtype=np.float32)
        else:
            logging.warning(f"Image file not found: {image_input}")
            return np.zeros(CFG.image_feature_dim, dtype=np.float32)
    elif isinstance(image_input, Image.Image):
        # PIL Image
        img_array = np.array(image_input, dtype=np.float32)
    elif isinstance(image_input, np.ndarray):
        # Direct numpy array
        img_array = image_input.astype(np.float32)
    else:
        logging.warning(f"Unknown image type: {type(image_input)}")
        return np.zeros(CFG.image_feature_dim, dtype=np.float32)
    
    # Flatten and normalize
    img_flat = img_array.flatten()
    
    # Normalize to [0, 1] if needed
    if img_flat.max() > 1.0:
        img_flat = img_flat / 255.0
    
    # Resize to target dimension
    target_dim = CFG.image_feature_dim
    if len(img_flat) > target_dim:
        # Downsample
        indices = np.linspace(0, len(img_flat)-1, target_dim, dtype=int)
        img_flat = img_flat[indices]
    elif len(img_flat) < target_dim:
        # Pad with zeros
        img_flat = np.pad(img_flat, (0, target_dim - len(img_flat)))
    
    return img_flat


def _convert_raw_action_to_ivntr_action(raw_action: Union[Dict, np.ndarray, List], env: BaseEnv) -> Action:
    """Convert your raw action format to IVNTR Action format."""
    action_dim = env.action_space.shape[0]
    action_array = np.zeros(action_dim, dtype=np.float32)
    
    if isinstance(raw_action, dict):
        # Dictionary format
        if 'joint_positions' in raw_action:
            joints = raw_action['joint_positions']
            action_array[:len(joints)] = joints
        elif 'pose' in raw_action:
            pose = raw_action['pose']
            action_array[:len(pose)] = pose
        elif 'values' in raw_action:
            values = raw_action['values']
            action_array[:len(values)] = values
    elif isinstance(raw_action, (np.ndarray, list)):
        # Direct array format
        raw_action = np.array(raw_action, dtype=np.float32)
        action_array[:len(raw_action)] = raw_action
    
    return Action(action_array)


# Example usage functions
def create_example_demonstration_file():
    """Create an example demonstration file showing the expected format."""
    demo_data = {
        'states': [
            {
                'point_cloud': np.random.random((1024, 3)).astype(np.float32),
                'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'object_poses': {
                    'block0': {'pose_x': 1.0, 'pose_y': 0.5, 'pose_z': 0.1, 'color_r': 0.8, 'color_g': 0.2, 'color_b': 0.1},
                    'robby': {'pose_x': 1.2, 'pose_y': 0.8, 'pose_z': 0.7, 'fingers': 1.0}
                }
            },
            {
                'point_cloud': np.random.random((1024, 3)).astype(np.float32),
                'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'object_poses': {
                    'block0': {'pose_x': 1.1, 'pose_y': 0.5, 'pose_z': 0.1, 'color_r': 0.8, 'color_g': 0.2, 'color_b': 0.1},
                    'robby': {'pose_x': 1.1, 'pose_y': 0.8, 'pose_z': 0.7, 'fingers': 0.0}
                }
            }
        ],
        'actions': [
            {'joint_positions': [0.1, 0.2, 0.3, 0.0]}  # Grasp action
        ]
    }
    
    # Save as pickle file
    os.makedirs('real_world_demos', exist_ok=True)
    with open('real_world_demos/demo_0.pkl', 'wb') as f:
        pickle.dump(demo_data, f)
    
    print("Created example demonstration file: real_world_demos/demo_0.pkl")