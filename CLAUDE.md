# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IVNTR (Inventory for Neural Predicate Discovery) is a research project for bilevel learning in robot planning. This is a neuro-symbolic bilevel learning framework where symbolic learning of predicate "effects" and neural learning of predicate "classifiers" alternate to learn neural predicates from demonstrations.

## Key Commands

### Setup and Installation
```bash
# Initial setup
git submodule update --init --recursive
pip install -e .
cd ext/downward && python build.py
mkdir -p saved_approaches saved_datasets logs
```

### Training
```bash
# Single domain training
bash scripts/train/satellites/satellites_biplan.sh
bash scripts/train/blocks_pos/blocks_pos_biplan.sh

# With LLM integration
bash scripts/train/satellites/satellites_biplan_llm.sh

# With PDDL integration
bash scripts/train/satellites/satellites_biplan_pddl.sh

# Parallel multi-group training (for large domains)
bash scripts/train/pickplace_stair/pickplace_biplan_up12.sh  # GPU0
bash scripts/train/pickplace_stair/pickplace_biplan_bp3.sh   # GPU1
```

### Testing/Evaluation
```bash
# Test trained models
bash scripts/test/satellites/satellites_ivntr.sh
bash scripts/test/blocks_pos/blocks_pos_ivntr.sh

# Test baselines
bash scripts/test/satellites/satellites_gnn_policy.sh
bash scripts/test/satellites/satellites_random.sh
```

### Main Entry Point
```bash
python predicators/main.py --env <environment> --approach <approach> --seed <seed>
```

### Code Quality
```bash
mypy predicators/  # Type checking
pylint predicators/  # Linting
```

## Architecture Overview

### Core Package Structure
- **`predicators/approaches/`** - Planning approaches including bilevel learning variants
- **`predicators/envs/`** - Environment implementations (satellites, blocks, tools, etc.)
- **`predicators/config/`** - YAML configuration files defining neural architectures and hyperparameters
- **`predicators/gnn/`** - Graph neural network and neural predicate invention modules
- **`predicators/llm/`** - Large language model integration modules
- **`predicators/nsrt_learning/`** - Neural state-relational template learning

### Key Design Patterns

**Approach System:** All planning approaches inherit from `BaseApproach` in `predicators/approaches/base_approach.py`. The main approach is `BilevelLearningApproach` with variants for LLM and PDDL integration.

**Configuration-Driven:** Each domain has YAML configs in `predicators/config/<domain>/` that define neural architecture parameters, training hyperparameters, and predicate group specifications.

**Multi-Modal Integration:** Combines neural components (GNNs, MLPs), symbolic components (PDDL planning), and LLM integration for predicate generation.

### Domain Mapping
- `satellites` - Satellite observation planning
- `blocks_pos` - Block stacking/manipulation  
- `view_plan_trivial` - Robot arm measurement tasks
- `view_plan_hard` - Complex robot navigation
- `pickplace_stair` - Object transportation
- `blocks_pcd` - Point cloud-based block manipulation

### Experiment Infrastructure
- **`scripts/train/`** - Training scripts organized by domain
- **`scripts/test/`** - Testing/evaluation scripts
- **`saved_approaches/`** - Trained models and neural predicate weights
- **`saved_datasets/`** - Generated training/testing datasets
- **`logs/`** - Experiment logs organized by domain and approach

### Important Environment Variables
Scripts automatically set:
- `FD_EXEC_PATH=ext/downward` - Fast Downward planner path
- `PYTHONHASHSEED=0` - Reproducible randomization
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` - GPU determinism

## Development Notes

### GPU Support
CUDA-optimized with PyTorch. Multi-GPU training supported for parallel predicate group learning. Device management is automatic with `device = "cuda:0"` in settings.

### Reproducibility
Comprehensive seeding across all random components with deterministic GPU operations. All experiments use structured logging and model checkpointing.

### External Dependencies
- **`ext/downward/`** - Fast Downward PDDL planner (git submodule, must be built)
- Uses Weights & Biases for experiment tracking
- MyPy configuration in `mypy.ini`