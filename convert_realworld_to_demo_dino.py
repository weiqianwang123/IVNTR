
#!/usr/bin/env python3
"""Convert real-world table cleaning data to demonstration format using DINO features."""

import json
import pickle
import dill
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from tqdm import tqdm
from gym.spaces import Box
from PIL import Image
import torch
from torchvision import transforms

from predicators import utils
from predicators.settings import CFG
from predicators.envs.clean_table_real import TableCleanRealEnv
from predicators.structs import (
    Action, LowLevelTrajectory, Dataset, State, Object, Type,
    ParameterizedOption, _Option
)


class RealWorldToDINOConverter:
    """Convert real-world table cleaning data to demo format with DINO features."""
    
    def __init__(self):
        """Initialize converter with TableCleanRealEnv settings and DINO model."""
        # Set required config
        CFG.seed = 0
        CFG.env = "clean-table-real-real"
        
        # Use the real environment with DINO features
        self.env = TableCleanRealEnv(use_gui=False)
        
        # Copy key attributes from environment
        self.dino_feature_dim = self.env.dino_feature_dim  # 1024
        
        # Environment parameters from TableCleanRealEnv
        self.center_x = 0.0
        self.center_y = 0.0
        self.side_x = 3.0
        self.side_y = 3.0
        
        # Get types from the environment (with DINO features)
        self._robot_type = self.env._robot_type
        self._toy_type = self.env._toy_type
        self._wiper_type = self.env._wiper_type
        self._box_type = self.env._box_type
        self._table_type = self.env._table_type
        
        # Use the environment's robot object
        self._robot = self.env._robot
        
        # Initialize DINO model
        self._init_dino_model()
        
        # Create parameterized options
        self._create_parameterized_options()
    
    def _init_dino_model(self):
        """Initialize DINO v3 model for feature extraction."""
        import sys
        sys.path.append("/home/qianwei/dinov3")
        
        try:
            # Try to import and load DINO directly
            from dinov3.models import vision_transformer as vit
            
            # Load checkpoint
            checkpoint_path = "/home/qianwei/IVNTR/predicators/config/clean_table_real/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
            self.dino_model = vit.vit_huge(patch_size=16)
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.dino_model.load_state_dict(state_dict['model'], strict=False)
            self.dino_model.eval()
        except Exception as e:
            print(f"Warning: Could not load DINO model directly: {e}")
            print("Using mock DINO features instead")
            self.dino_model = None
        
        # Preprocessing transform
        self.dino_transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
    
    def extract_dino_features(self, image_path: Path) -> np.ndarray:
        """Extract DINO features from an image."""
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}, using zeros")
            return np.zeros(self.dino_feature_dim, dtype=np.float32)
        
        # If DINO model failed to load, use random features
        if self.dino_model is None:
            # Use random features as placeholder
            np.random.seed(hash(str(image_path)) % 2**32)
            return np.random.randn(self.dino_feature_dim).astype(np.float32)
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.dino_transform(img).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.dino_model(img_tensor)  # Shape: (1, feature_dim)
        
        # Convert to numpy and flatten
        features_np = features.cpu().numpy().flatten()
        
        # Ensure correct dimensionality
        if len(features_np) != self.dino_feature_dim:
            print(f"Warning: Feature dimension mismatch. Expected {self.dino_feature_dim}, got {len(features_np)}")
            # Pad or truncate as needed
            if len(features_np) < self.dino_feature_dim:
                features_np = np.pad(features_np, (0, self.dino_feature_dim - len(features_np)))
            else:
                features_np = features_np[:self.dino_feature_dim]
        
        return features_np.astype(np.float32)
    
    def _create_parameterized_options(self):
        """Create ParameterizedOptions for all action types."""
        # Dummy policy, initiable, and terminal functions (not used during learning)
        def dummy_policy(s, m, o, p):
            return Action(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        
        def always_initiable(s, m, o, p):
            return True
        
        def never_terminal(s, m, o, p):
            return False
        def goal_terminal(s, m, o, p):
            # true when goal_achieved is set in the post-state
            for obj in s:
                if obj.type == self._robot_type:
                    return s.get(obj, "goal_achieved") > 0.5
            return False
        
        # Create ParameterizedOptions matching the environment's options
        self.options = {
            'PickToyFromTable': ParameterizedOption(
                'PickToyFromTable', 
                types=[self._robot_type, self._toy_type,self._table_type],
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
                types=[self._robot_type, self._wiper_type,self._table_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PlaceWiperAtTable': ParameterizedOption(
                'PlaceWiperAtTable',
                types=[self._robot_type, self._wiper_type,self._table_type],
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
                types=[self._robot_type, self._box_type,self._table_type],
                params_space=Box(0, 1, (0,), dtype=np.float32),
                policy=dummy_policy,
                initiable=always_initiable,
                terminal=never_terminal
            ),
            'PullBoxIn': ParameterizedOption(
                'PullBoxIn',
                types=[self._robot_type, self._box_type,self._table_type],
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
                terminal=goal_terminal
            )
        }
    
    def parse_action_string(self, action_str: str) -> Tuple[str, List[str]]:
        """Parse action string to get action name and arguments."""
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
        """Convert a single trajectory from raw data to demo format with DINO features."""
        print(f"Converting trajectory from {traj_dir.name} with DINO features...")
        
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
            
            # For the final two states, use previous state's images for objects
            # This is because the final states might not have corresponding images
            if state_idx >= num_states - 1:
                img_state_idx = num_states - 2  # Use the third-to-last state's images
            else:
                img_state_idx = state_idx
            
            # Robot state with DINO features and goal_achieved
            robot_img_path = traj_dir / f"robot_1" / f"state_{img_state_idx}_cropped.png"
            if robot_img_path.exists():
                robot_dino_features = self.extract_dino_features(robot_img_path)
            else:
                # Robot might not have images, use zeros
                robot_dino_features = np.zeros(self.dino_feature_dim, dtype=np.float32)
            
            state_dict[robot] = {
                "goal_achieved": 1.0 if state_idx == num_states-1 else 0.0,
                **{f"dino_{j}": robot_dino_features[j] for j in range(self.dino_feature_dim)}
            }
            
            # Table state with DINO features
            table_img_path = traj_dir / f"table_1" / f"state_{img_state_idx}_cropped.png"
            table_dino_features = self.extract_dino_features(table_img_path)
            state_dict[table] = {
                **{f"dino_{j}": table_dino_features[j] for j in range(self.dino_feature_dim)}
            }
            
            # Box state with DINO features
            box_img_path = traj_dir / f"box_1" / f"state_{img_state_idx}_cropped.png"
            box_dino_features = self.extract_dino_features(box_img_path)
            state_dict[box] = {
                **{f"dino_{j}": box_dino_features[j] for j in range(self.dino_feature_dim)}
            }
            
            # Toy states with DINO features
            for j, toy in enumerate(toys):
                toy_img_path = traj_dir / f"toy_{j+1}" / f"state_{img_state_idx}_cropped.png"
                toy_dino_features = self.extract_dino_features(toy_img_path)
                state_dict[toy] = {
                    **{f"dino_{j}": toy_dino_features[j] for j in range(self.dino_feature_dim)}
                }
            
            # Wiper states with DINO features
            for j, wiper in enumerate(wipers):
                wiper_img_path = traj_dir / f"wiper_{j+1}" / f"state_{img_state_idx}_cropped.png"
                wiper_dino_features = self.extract_dino_features(wiper_img_path)
                state_dict[wiper] = {
                    **{f"dino_{j}": wiper_dino_features[j] for j in range(self.dino_feature_dim)}
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
                        elif action_name == 'PickToyFromTable':
                            objects.append(table)
                    elif action_name in ['PickWiperFromBox', 'PickWiperFromTable', 
                                        'PlaceWiperAtTable', 'PlaceWiperToBox', 'WipeTable']:
                        objects.append(wipers[0])
                        if action_name == 'PickWiperFromBox':
                            objects.append(box)
                        elif action_name == 'PlaceWiperToBox':
                            objects.append(box)
                        else :
                            objects.append(table)
                    elif action_name in ['PushBoxOut', 'PullBoxIn']:
                        objects.append(box)
                        objects.append(table)
                    elif action_name == 'AchieveGoal':
                        objects.append(table)
                    
                    # Create and attach the option
                    option = param_option.ground(objects, np.array([], dtype=np.float32))
                    action.set_option(option)
                print(f"  Created action: {action_str} as {action}")
                actions.append(action)
        
        print(f"  Successfully converted trajectory {traj_idx+1}: {len(states)} states, {len(actions)} actions")
        return LowLevelTrajectory(states, actions, _is_demo=True, _train_task_idx=0)
    
    def convert_dataset(self, data_dir: Path, num_demos: int = None) -> Dataset:
        """Convert real-world data to Dataset format with DINO features."""
        # Find all trajectory directories
        traj_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        if num_demos is not None:
            traj_dirs = traj_dirs[:num_demos]
        
        print(f"Found {len(traj_dirs)} trajectory directories to convert")
        
        # Convert each trajectory
        trajectories = []
        for idx, traj_dir in tqdm(enumerate(traj_dirs), desc="Converting trajectories"):
            try:
                traj = self.convert_trajectory(traj_dir, idx)
                trajectories.append(traj)
            except Exception as e:
                print(f"Error converting {traj_dir}: {e}")
                continue
        
        return Dataset(trajectories)


def main():
    parser = argparse.ArgumentParser(description='Convert real-world data to demo format with DINO features')
    parser.add_argument('--data-dir', type=str, 
                       default='/home/qianwei/IVNTR/saved_raw_data_real/table-clean',
                       help='Path to raw data directory')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/qianwei/IVNTR/saved_datasets',
                       help='Output directory for converted datasets')
    parser.add_argument('--num-demos', type=int, default=None,
                       help='Number of demonstrations to convert (None for all)')
    
    args = parser.parse_args()
    
    # Initialize converter
    print("Initializing converter with DINO model...")
    converter = RealWorldToDINOConverter()
    
    # Convert dataset
    print("Converting trajectories...")
    dataset = converter.convert_dataset(Path(args.data_dir), args.num_demos)
    
    # Save dataset
    num_demos = args.num_demos if args.num_demos else len(dataset.trajectories)
    output_file = Path(args.output_dir) / f"clean-table-real-real__demo__realworld__{num_demos}____0__None.data"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'wb') as f:
        dill.dump(dataset, f)
    
    print(f"\nConversion complete!")
    print(f"Created dataset with {len(dataset.trajectories)} trajectories")
    print(f"Output file: {output_file}")
    
    # Print summary statistics
    avg_states = sum(len(t.states) for t in dataset.trajectories) / len(dataset.trajectories)
    avg_actions = sum(len(t.actions) for t in dataset.trajectories) / len(dataset.trajectories)
    print(f"Average states per trajectory: {avg_states:.1f}")
    print(f"Average actions per trajectory: {avg_actions:.1f}")


if __name__ == "__main__":
    main()