# Majority Voting Precondition Learning Implementation

## Summary of Changes

I've implemented majority voting as an alternative to intersection-based precondition learning in the IVNTR bilevel learning framework.

## Files Modified

### 1. `predicators/settings.py`
- Added `strips_learner_precondition_method` = "intersection" | "majority_vote"
- Added `strips_learner_majority_threshold` = 0.5 (minimum fraction of segments)

### 2. `predicators/nsrt_learning/strips_learning/base_strips_learner.py`
- Added `_induce_preconditions_via_majority_vote()` method
- Added `_induce_preconditions()` unified method that chooses based on config
- Keeps original `_induce_preconditions_via_intersection()` for backward compatibility

### 3. Updated all STRIPS learners to use the unified method:
- `belief_learner.py`
- `clustering_learner.py` 
- `pnad_search_learner.py`
- `gen_to_spec_learner.py`
- `oracle_clustering_learner.py`

## How It Works

### Original Intersection Method
```python
# Only atoms present in ALL segments become preconditions
for each segment:
    if first_segment:
        preconditions = segment_atoms
    else:
        preconditions = preconditions ∩ segment_atoms  # INTERSECTION
```

### New Majority Voting Method
```python
# Count atom occurrences across all segments
atom_counts = {}
for each segment:
    for atom in segment_atoms:
        atom_counts[atom] += 1

# Include atoms that appear in ≥ threshold fraction of segments
min_count = threshold * total_segments
preconditions = {atom for atom, count in atom_counts.items() 
                 if count >= min_count}
```

## Benefits

1. **Robustness to noise**: Missing conditions in a few demonstrations won't eliminate important preconditions
2. **Configurable strictness**: Threshold parameter allows tuning between conservative and inclusive learning
3. **Better recall**: More likely to capture true preconditions that were accidentally omitted in some demos
4. **Backward compatible**: Original intersection method still available as default

## Usage

Enable majority voting in your config or with command line args:
```python
# In config file or settings
strips_learner_precondition_method = "majority_vote"
strips_learner_majority_threshold = 0.75  # 75% of segments must contain atom
```

## Example

Given 4 segments for "PickUp" operator:
- Segment 0: {HandEmpty, OnTable, Clear}
- Segment 1: {HandEmpty, OnTable, Clear} 
- Segment 2: {HandEmpty, OnTable, Clear}
- Segment 3: {HandEmpty, OnTable}  # Missing Clear!

**Intersection result**: {HandEmpty, OnTable} - Clear eliminated!
**Majority vote (75% threshold)**: {HandEmpty, OnTable, Clear} - Clear included!

The majority voting approach would correctly learn that "Clear" is likely a necessary precondition even though it was missing from one noisy demonstration.