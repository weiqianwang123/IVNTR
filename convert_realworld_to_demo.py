#!/usr/bin/env python3
"""
Convert real-world data to demonstration trajectory format.

This script processes the collected real-world data and converts it to the format
needed for predicate learning in bilevel_learning_pdlm_approach.
"""

import json
import os
import dill as pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add predicators to Python path
sys.path.append('/home/qianwei/IVNTR')

# Set required configuration before importing
from predicators import settings
settings.CFG.seed = 0
settings.CFG.num_train_tasks = 60
settings.CFG.num_test_tasks = 10

from predicators.structs import State, Action, LowLevelTrajectory, Dataset, Object, Type, ParameterizedOption, _Option
from predicators import utils
from gym.spaces import Box


class RealWorldDataConverter:
    """Convert real-world collected data to demonstration format using object-centric images."""
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize environment to get the actual objects and types
        # This ensures object instances match what predicates expect
        from predicators.envs.clean_table_real import TableCleanEnv
        self.env = TableCleanEnv(use_gui=False)
        
        # Store dimensions from environment
        self.img_height = self.env.img_height
        self.img_width = self.env.img_width
        self.img_channels = self.env.img_channels
        self.img_size = self.env.img_size
        self.center_x = self.env.center_x
        self.center_y = self.env.center_y
        self.side_x = self.env.side_x
        self.side_y = self.env.side_y
        
        # Use the actual types from the environment
        self._robot_type = self.env._robot_type
        self._toy_type = self.env._toy_type
        self._wiper_type = self.env._wiper_type
        self._box_type = self.env._box_type
        self._table_type = self.env._table_type
        
        # Use the actual robot object from the environment
        self._robot = self.env._robot
        
        # Create ParameterizedOptions for each action type
        self._create_parameterized_options()
    
    def _create_parameterized_options(self):
        """Create ParameterizedOptions for all action types."""
        # Dummy policy, initiable, and terminal functions (not used during learning)
        def dummy_policy(s, m, o, p):
            return Action(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        
        def always_initiable(s, m, o, p):
            return True
        
        def never_terminal(s, m, o, p):
            return False
        
        # Create ParameterizedOptions matching the environment's options
        self.options = {
            'PickToyFromTable': ParameterizedOption(
                'PickToyFromTable', 
                types=[self._robot_type, self._toy_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PlaceToyToBox': ParameterizedOption(
                'PlaceToyToBox',
                types=[self._robot_type, self._toy_type, self._box_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PickWiperFromBox': ParameterizedOption(
                'PickWiperFromBox',
                types=[self._robot_type, self._wiper_type, self._box_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PickWiperFromTable': ParameterizedOption(
                'PickWiperFromTable',
                types=[self._robot_type, self._wiper_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PlaceWiperAtTable': ParameterizedOption(
                'PlaceWiperAtTable',
                types=[self._robot_type, self._wiper_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PlaceWiperToBox': ParameterizedOption(
                'PlaceWiperToBox',
                types=[self._robot_type, self._wiper_type, self._box_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PushBoxOut': ParameterizedOption(
                'PushBoxOut',
                types=[self._robot_type, self._box_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PullBoxIn': ParameterizedOption(
                'PullBoxIn',
                types=[self._robot_type, self._box_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'WipeTable': ParameterizedOption(
                'WipeTable',
                types=[self._robot_type, self._wiper_type, self._table_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'AchieveGoal': ParameterizedOption(
                'AchieveGoal',
                types=[self._robot_type, self._table_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            )
        }
    
    def load_image_as_features(self, image_path: Path) -> np.ndarray:
        """Load object-centric image and flatten to feature vector."""
        if image_path.exists():
            # Load image and resize to expected dimensions
            img = Image.open(image_path).convert("RGB")
            img = img.resize((self.img_width, self.img_height))
            
            # Convert to numpy array and flatten
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            return img_array.flatten()
        else:
            # Return zeros if image not found
            print(f"Warning: Image not found: {image_path}")
            return np.zeros(self.img_size, dtype=np.float32)
    
    def parse_action_string(self, action_str: str) -> Tuple[str, List[str]]:
        """Parse action string like 'PullBoxIn(robot1, box1)' into name and args."""
        if '(' not in action_str:
            return action_str, []
        
        action_name = action_str.split('(')[0]
        args_str = action_str.split('(')[1].rstrip(')')
        args = [arg.strip() for arg in args_str.split(',')]
        return action_name, args
    
    def action_name_to_type(self, action_name: str) -> int:
        """Convert action name to action type integer."""
        action_map = {
            'PickToyFromTable': 0,
            'PlaceToyToBox': 1,
            'PickWiperFromBox': 2,
            'PickWiperFromTable': 3,
            'PlaceWiperAtTable': 4,
            'PlaceWiperToBox': 5,
            'PushBoxOut': 6,
            'PullBoxIn': 7,
            'WipeTable': 8,
            'AchieveGoal': 9
        }
        return action_map.get(action_name, -1)
    
    def convert_trajectory(self, traj_dir: Path, traj_idx: int) -> LowLevelTrajectory:
        """Convert a single trajectory from raw data to demo format."""
        print(f"Converting trajectory from {traj_dir.name}...")
        
        # Load config
        config_file = traj_dir / 'config.json'
        with open(config_file) as f:
            config_data = json.load(f)
        
        # Extract object counts
        num_toys = config_data['object_config'].get('toy', 2)
        num_wipers = config_data['object_config'].get('wiper', 1)
        num_boxes = config_data['object_config'].get('box', 1)
        
        # Use the actual robot object from environment, create others
        robot = self._robot  # Use the environment's robot object
        table = Object("table0", self._table_type)
        box = Object("box0", self._box_type)
        
        toys = [Object(f"toy{i}", self._toy_type) for i in range(num_toys)]
        wipers = [Object(f"wiper{i}", self._wiper_type) for i in range(num_wipers)]
        
        # Parse action sequence
        action_sequence_file = traj_dir / 'action_sequence.txt'
        action_sequence = []
        with open(action_sequence_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    action_sequence.append(line)
        
        # Add an additional AchieveGoal to ensure there are two
        # (one from file, one extra)
        # action_sequence.append('AchieveGoal(robot1, table1)')
        
        # Determine number of states (actions + 1 for final state)
        num_states = len(action_sequence) + 1
        
        states = []
        actions = []
        
        # Process each state
        for state_idx in range(num_states):
            # Create state dictionary
            state_dict = {}
            
            # For the final state, use previous state's images for objects (except robot)
            # This is because the final state might not have corresponding images
            img_state_idx = min(state_idx, num_states - 1) if state_idx == num_states - 1 else state_idx
            
            # Robot state (only basic features, no image)
            state_dict[robot] = {
                "handempty": 1.0,  # Simplified assumption
                "goal_achieved": 1.0 if state_idx == num_states-1  else 0.0
            }
            
            # Table state with image features
            table_img_path = traj_dir / f"table_1" / f"state_{img_state_idx}_cropped.png"
            table_img_features = self.load_image_as_features(table_img_path)
            state_dict[table] = {
                "pose_x": 0.0,
                "pose_y": 0.0,
                "is_clean": 1.0 if state_idx >= num_states - 2 else 0.0,
                **{f"img_{j}": table_img_features[j] for j in range(self.img_size)}
            }
            
            # Box state with image features
            box_img_path = traj_dir / f"box_1" / f"state_{img_state_idx}_cropped.png"
            box_img_features = self.load_image_as_features(box_img_path)
            # Estimate box position based on action progress
            box_at_center = state_idx < 4  # Box moves out after initial actions
            state_dict[box] = {
                "pose_x": self.center_x if box_at_center else self.side_x,
                "pose_y": self.center_y if box_at_center else self.side_y,
                "at_center": 1.0 if box_at_center else 0.0,
                "at_side": 0.0 if box_at_center else 1.0,
                **{f"img_{j}": box_img_features[j] for j in range(self.img_size)}
            }
            
            # Toy states with image features
            for j, toy in enumerate(toys):
                toy_img_path = traj_dir / f"toy_{j+1}" / f"state_{img_state_idx}_cropped.png"
                toy_img_features = self.load_image_as_features(toy_img_path)
                # Estimate toy position based on action progress
                toy_on_table = state_idx < (5 + j * 2)  # Toys picked up in sequence
                state_dict[toy] = {
                    "pose_x": 0.0 if toy_on_table else self.center_x,
                    "pose_y": 0.0 if toy_on_table else self.center_y,
                    "on_table": 1.0 if toy_on_table else 0.0,
                    "in_box": 0.0 if toy_on_table else 1.0,
                    **{f"img_{j}": toy_img_features[j] for j in range(self.img_size)}
                }
            
            # Wiper states with image features
            for j, wiper in enumerate(wipers):
                wiper_img_path = traj_dir / f"wiper_{j+1}" / f"state_{img_state_idx}_cropped.png"
                wiper_img_features = self.load_image_as_features(wiper_img_path)
                # Estimate wiper position based on action progress
                wiper_in_box = state_idx < 2
                wiper_on_table = 2 <= state_idx < num_states - 2
                state_dict[wiper] = {
                    "pose_x": self.center_x if wiper_in_box else 0.0,
                    "pose_y": self.center_y if wiper_in_box else 0.0,
                    "on_table": 1.0 if wiper_on_table else 0.0,
                    "in_box": 1.0 if wiper_in_box else 0.0,
                    **{f"img_{j}": wiper_img_features[j] for j in range(self.img_size)}
                }
            
            # Create state
            state = utils.create_state_from_dict(state_dict)
            states.append(state)
            
            # Create action (if not the last state)
            if state_idx < len(action_sequence):
                action_str = action_sequence[state_idx]
                action_name, args = self.parse_action_string(action_str)
                action_type = self.action_name_to_type(action_name)
                
                # Create action with blank x,y parameters as requested
                # Action format: [action_type, x, y]
                action = Action(np.array([action_type, 0.0, 0.0], dtype=np.float32))
                
                # Attach the corresponding option to the action
                if action_name in self.options:
                    param_option = self.options[action_name]
                    
                    # Determine objects based on action type
                    objects = [robot]  # Robot is always first
                    
                    if action_name in ['PickToyFromTable', 'PlaceToyToBox']:
                        # Find which toy is involved (simplified: use first toy)
                        objects.append(toys[0])
                        if action_name == 'PlaceToyToBox':
                            objects.append(box)
                    elif action_name in ['PickWiperFromBox', 'PickWiperFromTable', 
                                        'PlaceWiperAtTable', 'PlaceWiperToBox', 'WipeTable']:
                        objects.append(wipers[0])
                        if action_name == 'PickWiperFromBox':
                            objects.append(box)
                        elif action_name == 'PlaceWiperToBox':
                            objects.append(box)
                        elif action_name == 'WipeTable':
                            objects.append(table)
                    elif action_name in ['PushBoxOut', 'PullBoxIn']:
                        objects.append(box)
                    elif action_name == 'AchieveGoal':
                        objects.append(table)
                    
                    # Create and attach the option
                    option = param_option.ground(objects, np.array([], dtype=np.float32))
                    action.set_option(option)
                print(f"  Created action: {action_str} as {action}")
                actions.append(action)




        
        # Create trajectory marked as demo
        trajectory = LowLevelTrajectory(states, actions, _is_demo=True, _train_task_idx=0)
        return trajectory
    
    def convert_all_trajectories(self, num_demos: int = None) -> Dataset:
        """Convert all trajectories in the raw data directory.
        
        Args:
            num_demos: Number of demos to convert (None for all)
        """
        trajectories = []
        
        # Find all trajectory directories
        table_clean_dir = self.raw_data_dir / 'table-clean'
        if not table_clean_dir.exists():
            raise ValueError(f"Directory not found: {table_clean_dir}")
            
        traj_dirs = []
        for item in sorted(table_clean_dir.iterdir()):
            if item.is_dir() and item.name.isdigit():
                traj_dirs.append(item)
        
        traj_dirs = sorted(traj_dirs, key=lambda x: int(x.name))
        
        if num_demos is not None:
            traj_dirs = traj_dirs[:num_demos]
        
        print(f"Found {len(traj_dirs)} trajectory directories to convert")
        
        for traj_dir in tqdm(traj_dirs, desc="Converting trajectories"):
            traj_idx = int(traj_dir.name)
            try:
                trajectory = self.convert_trajectory(traj_dir, traj_idx)
                trajectories.append(trajectory)
                print(f"  Successfully converted trajectory {traj_idx}: "
                      f"{len(trajectory.states)} states, {len(trajectory.actions)} actions")
            except Exception as e:
                print(f"  Error converting trajectory {traj_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return Dataset(trajectories)
    
    def save_dataset(self, dataset: Dataset, filename: str):
        """Save dataset to file."""
        output_path = self.output_dir / filename
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {output_path}")


def main():
    """Main conversion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert real-world data to demonstration format")
    parser.add_argument("--raw-data-dir", type=str, 
                       default="/home/qianwei/IVNTR/saved_raw_data_real",
                       help="Directory containing raw real-world data")
    parser.add_argument("--output-dir", type=str,
                       default="/home/qianwei/IVNTR/saved_datasets", 
                       help="Output directory for converted dataset")
    parser.add_argument("--num-demos", type=int, default=None,
                       help="Number of demos to convert (default: all)")
    parser.add_argument("--output-name", type=str, default=None,
                       help="Custom output filename (default: auto-generated)")
    
    args = parser.parse_args()
    
    print("Initializing converter...")
    converter = RealWorldDataConverter(args.raw_data_dir, args.output_dir)
    
    # Convert trajectories
    print("Converting trajectories...")
    dataset = converter.convert_all_trajectories(num_demos=args.num_demos)
    
    # Generate filename
    num_trajs = len(dataset.trajectories)
    if args.output_name:
        filename = args.output_name
    else:
        filename = f"clean-table-real__demo__realworld__{num_trajs}____0__None.data"
    
    # Save dataset
    converter.save_dataset(dataset, filename)
    
    print(f"\nConversion complete!")
    print(f"Created dataset with {num_trajs} trajectories")
    print(f"Output file: {args.output_dir}/{filename}")
    
    # Print summary statistics
    if num_trajs > 0:
        avg_states = sum(len(t.states) for t in dataset.trajectories) / num_trajs
        avg_actions = sum(len(t.actions) for t in dataset.trajectories) / num_trajs
        print(f"Average states per trajectory: {avg_states:.1f}")
        print(f"Average actions per trajectory: {avg_actions:.1f}")


if __name__ == "__main__":
    main()