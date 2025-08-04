"""Utility script to prepare real-world demonstration data for IVNTR integration.

This script helps convert your existing state-action-state-action demonstrations
into the format expected by IVNTR.
"""

import argparse
import os
import pickle
import json
import numpy as np
from typing import Dict, List, Any, Union
from PIL import Image


def prepare_demonstration_data(input_dir: str, output_dir: str, demo_format: str = 'auto'):
    """Convert your demonstration data to IVNTR-compatible format.
    
    Args:
        input_dir: Directory containing your original demonstration files
        output_dir: Directory to save IVNTR-compatible files
        demo_format: Format of your demonstrations ('auto', 'pickle', 'json', 'npz')
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find demonstration files
    demo_files = []
    for ext in ['.pkl', '.json', '.npz'] if demo_format == 'auto' else [f'.{demo_format}']:
        demo_files.extend([f for f in os.listdir(input_dir) if f.endswith(ext)])
    
    demo_files.sort()
    
    print(f"Found {len(demo_files)} demonstration files")
    
    for i, demo_file in enumerate(demo_files):
        print(f"Processing {demo_file}...")
        
        input_path = os.path.join(input_dir, demo_file)
        output_path = os.path.join(output_dir, f"demo_{i}.pkl")
        
        # Convert demonstration
        converted_demo = convert_demonstration_file(input_path)
        
        if converted_demo is not None:
            # Save in standard format
            with open(output_path, 'wb') as f:
                pickle.dump(converted_demo, f)
            print(f"  -> Saved as demo_{i}.pkl")
        else:
            print(f"  -> Failed to convert {demo_file}")


def convert_demonstration_file(file_path: str) -> Dict[str, Any]:
    """Convert a single demonstration file to IVNTR format.
    
    Expected IVNTR format:
    {
        'states': [
            {
                'point_cloud': np.array (N, 3),
                'image': np.array (H, W, C) or PIL Image,
                'object_poses': {
                    'object_name': {'pose_x': x, 'pose_y': y, ...}
                }
            }, ...
        ],
        'actions': [
            {'joint_positions': [...]}, ...
        ]
    }
    """
    
    # Load your demonstration data
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.endswith('.npz'):
        data = dict(np.load(file_path, allow_pickle=True))
    else:
        print(f"Unsupported file format: {file_path}")
        return None
    
    # Convert to IVNTR format
    # YOU NEED TO MODIFY THIS SECTION BASED ON YOUR DATA FORMAT
    
    if 'trajectory' in data:
        # Example: If your data has a 'trajectory' key
        return convert_trajectory_format(data['trajectory'])
    elif 'states' in data and 'actions' in data:
        # Already in correct format
        return validate_and_fix_format(data)
    else:
        # Custom conversion based on your specific format
        return convert_custom_format(data)


def convert_trajectory_format(trajectory_data: Dict) -> Dict[str, Any]:
    """Convert trajectory format to IVNTR format.
    
    Modify this function based on your specific data structure.
    """
    
    converted = {
        'states': [],
        'actions': []
    }
    
    # Example conversion (MODIFY THIS):
    # Assuming your trajectory has 'observations', 'actions', 'point_clouds', 'images'
    
    observations = trajectory_data.get('observations', [])
    actions = trajectory_data.get('actions', [])
    point_clouds = trajectory_data.get('point_clouds', [])
    images = trajectory_data.get('images', [])
    
    # Convert states
    for i, obs in enumerate(observations):
        state = {
            'point_cloud': point_clouds[i] if i < len(point_clouds) else None,
            'image': images[i] if i < len(images) else None,
            'object_poses': extract_object_poses_from_observation(obs)
        }
        converted['states'].append(state)
    
    # Convert actions
    for action in actions:
        converted_action = convert_action_format(action)
        converted['actions'].append(converted_action)
    
    return converted


def extract_object_poses_from_observation(obs: Any) -> Dict[str, Dict[str, float]]:
    """Extract object poses from your observation format.
    
    MODIFY THIS based on how your observations are structured.
    """
    
    object_poses = {}
    
    # Example extraction (MODIFY THIS):
    if isinstance(obs, dict):
        for obj_name, obj_data in obs.items():
            if isinstance(obj_data, dict):
                object_poses[obj_name] = {
                    'pose_x': obj_data.get('x', obj_data.get('pos_x', 0.0)),
                    'pose_y': obj_data.get('y', obj_data.get('pos_y', 0.0)),
                    'pose_z': obj_data.get('z', obj_data.get('pos_z', 0.0)),
                    'color_r': obj_data.get('color', [0.5, 0.5, 0.5])[0],
                    'color_g': obj_data.get('color', [0.5, 0.5, 0.5])[1],
                    'color_b': obj_data.get('color', [0.5, 0.5, 0.5])[2],
                    'goal_achieved': obj_data.get('goal_achieved', 0.0),
                    'fingers': obj_data.get('gripper', obj_data.get('fingers', 1.0))
                }
    
    return object_poses


def convert_action_format(action: Any) -> Dict[str, Any]:
    """Convert your action format to IVNTR format.
    
    MODIFY THIS based on your action representation.
    """
    
    if isinstance(action, dict):
        return action  # Already in dict format
    elif isinstance(action, (list, np.ndarray)):
        return {'joint_positions': list(action)}
    else:
        print(f"Unknown action format: {type(action)}")
        return {'joint_positions': [0.0] * 7}  # Default 7-DOF action


def convert_custom_format(data: Dict) -> Dict[str, Any]:
    """Convert your custom data format to IVNTR format.
    
    MODIFY THIS ENTIRE FUNCTION based on your specific data structure.
    """
    
    print("Converting custom format - please modify this function!")
    print(f"Available keys in data: {list(data.keys())}")
    
    # Template conversion
    converted = {
        'states': [],
        'actions': []
    }
    
    # Example: If your data structure is different
    # Replace this with your actual conversion logic
    
    return converted


def validate_and_fix_format(data: Dict) -> Dict[str, Any]:
    """Validate and fix format issues in demonstration data."""
    
    states = data.get('states', [])
    actions = data.get('actions', [])
    
    # Ensure correct state-action relationship
    if len(states) != len(actions) + 1:
        print(f"Warning: State-action length mismatch: {len(states)} states, {len(actions)} actions")
        # Fix by truncating
        min_len = min(len(states), len(actions) + 1)
        states = states[:min_len]
        actions = actions[:min_len-1]
    
    # Validate each state has required fields
    for i, state in enumerate(states):
        if 'point_cloud' not in state:
            print(f"Warning: State {i} missing point_cloud")
            state['point_cloud'] = np.zeros((1024, 3), dtype=np.float32)
        
        if 'image' not in state:
            print(f"Warning: State {i} missing image")
            state['image'] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if 'object_poses' not in state:
            print(f"Warning: State {i} missing object_poses")
            state['object_poses'] = {}
    
    return {'states': states, 'actions': actions}


def main():
    parser = argparse.ArgumentParser(description='Prepare real-world demonstration data for IVNTR')
    parser.add_argument('--input_dir', required=True, help='Input directory with your demonstration files')
    parser.add_argument('--output_dir', default='real_world_demos', help='Output directory for IVNTR format')
    parser.add_argument('--format', choices=['auto', 'pickle', 'json', 'npz'], default='auto',
                       help='Format of your demonstration files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    print("IMPORTANT: This script contains template conversion functions.")
    print("You MUST modify the conversion functions based on your specific data format.")
    print("Look for comments marked with 'MODIFY THIS' in the code.")
    print()
    
    prepare_demonstration_data(args.input_dir, args.output_dir, args.format)
    
    print(f"\nConversion complete! Demonstration files saved to: {args.output_dir}")
    print("\nTo use with IVNTR, run:")
    print("python predicators/main.py --env your_env --approach ivntr \\")
    print("  --offline_data_method direct_real_world_demo --seed 0")


if __name__ == "__main__":
    main()