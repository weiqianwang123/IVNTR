#!/usr/bin/env python3
"""Inspect the contents of a dataset file."""

import sys
import pickle
import dill
import numpy as np
from pathlib import Path

def inspect_dataset(filepath):
    """Inspect and print details of a dataset file."""
    
    # Try both pickle and dill
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print("Loaded with pickle")
    except:
        try:
            with open(filepath, 'rb') as f:
                data = dill.load(f)
            print("Loaded with dill")
        except Exception as e:
            print(f"Error loading file: {e}")
            return
    
    print("=" * 80)
    print(f"DATASET: {filepath}")
    print("=" * 80)
    
    print(f"\nDataset type: {type(data)}")
    print(f"Number of trajectories: {len(data.trajectories)}")
    
    for i, traj in enumerate(data.trajectories):
        print(f"\n{'='*60}")
        print(f"TRAJECTORY {i+1}")
        print(f"{'='*60}")
        
        print(f"  Number of states: {len(traj.states)}")
        print(f"  Number of actions: {len(traj.actions)}")
        print(f"  Is demo: {traj.is_demo}")
        print(f"  Train task idx: {traj.train_task_idx}")
        
        # First state details
        print(f"\n  First State:")
        state = traj.states[0]
        print(f"    Objects in state: {len(state.data)}")
        for obj in sorted(state.data.keys(), key=lambda x: x.name):
            print(f"      - {obj.name} (type: {obj.type.name})")
            # Get features
            obj_features = []
            for feat_name in obj.type.feature_names[:5]:  # Show first 5 features
                if hasattr(state, 'get'):
                    val = state.get(obj, feat_name)
                    obj_features.append(f"{feat_name}={val:.2f}" if isinstance(val, float) else f"{feat_name}={val}")
            if obj_features:
                print(f"        Features: {', '.join(obj_features)}")
            # Check for image features
            img_count = sum(1 for fn in obj.type.feature_names if fn.startswith('img_'))
            if img_count > 0:
                print(f"        Image features: {img_count} pixels")
        
        # Last state summary
        print(f"\n  Last State:")
        state = traj.states[-1]
        for obj in sorted(state.data.keys(), key=lambda x: x.name):
            # Show key features for last state
            key_features = []
            if 'goal_achieved' in obj.type.feature_names:
                val = state.get(obj, 'goal_achieved')
                key_features.append(f"goal_achieved={val}")
            if 'is_clean' in obj.type.feature_names:
                val = state.get(obj, 'is_clean')
                key_features.append(f"is_clean={val}")
            if 'on_table' in obj.type.feature_names:
                val = state.get(obj, 'on_table')
                key_features.append(f"on_table={val}")
            if 'in_box' in obj.type.feature_names:
                val = state.get(obj, 'in_box')
                key_features.append(f"in_box={val}")
            
            if key_features:
                print(f"      {obj.name}: {', '.join(key_features)}")
        
        # Actions summary
        print(f"\n  Actions:")
        for j, action in enumerate(traj.actions[:5]):  # Show first 5 actions
            print(f"    [{j}] Action array: {action.arr}")
            if action.has_option():
                option = action.get_option()
                print(f"        Option: {option.name}")
                print(f"        Objects: {[obj.name for obj in option.objects]}")
                print(f"        Params: {option.params}")
        
        if len(traj.actions) > 5:
            print(f"    ... ({len(traj.actions) - 5} more actions)")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if len(data.trajectories) > 0:
        avg_states = sum(len(t.states) for t in data.trajectories) / len(data.trajectories)
        avg_actions = sum(len(t.actions) for t in data.trajectories) / len(data.trajectories)
        print(f"Average states per trajectory: {avg_states:.1f}")
        print(f"Average actions per trajectory: {avg_actions:.1f}")
        
        # Check if all actions have options
        total_actions = sum(len(t.actions) for t in data.trajectories)
        actions_with_options = sum(
            sum(1 for a in t.actions if a.has_option()) 
            for t in data.trajectories
        )
        print(f"Actions with options: {actions_with_options}/{total_actions}")
        
        # Count unique object types
        all_types = set()
        for traj in data.trajectories:
            for state in traj.states:
                for obj in state.data.keys():
                    all_types.add(obj.type.name)
        print(f"Object types: {sorted(all_types)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "saved_datasets/clean-table-real__demo__oracle__7____0__None.data"
    
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        sys.exit(1)
    
    inspect_dataset(filepath)