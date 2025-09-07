#!/usr/bin/env python3
"""
Example demonstrating the difference between intersection and majority voting
for precondition learning in IVNTR.
"""

def demonstrate_precondition_learning():
    """Show how intersection vs majority voting works with example data."""
    
    print("=== PRECONDITION LEARNING COMPARISON ===\n")
    
    # Example: PickUp operator observed in 4 segments
    segments = [
        {"HandEmpty", "OnTable", "Clear"},           # Segment 0
        {"HandEmpty", "OnTable", "Clear"},           # Segment 1  
        {"HandEmpty", "OnTable", "Clear"},           # Segment 2
        {"HandEmpty", "OnTable"},                    # Segment 3 (missing Clear)
    ]
    
    print("Operator 'PickUp' observed in 4 segments:")
    for i, atoms in enumerate(segments):
        print(f"  Segment {i}: {atoms}")
    
    # Intersection method (current default)
    intersection_result = set.intersection(*[set(s) for s in segments])
    print(f"\nINTERSECTION METHOD:")
    print(f"  Result: {intersection_result}")
    print(f"  Only atoms present in ALL segments become preconditions")
    print(f"  'Clear' eliminated because missing from 1 out of 4 segments!")
    
    # Majority voting method (new implementation)
    from collections import Counter
    
    def majority_vote(segments, threshold=0.5):
        atom_counts = Counter()
        total_segments = len(segments)
        
        # Count each atom across all segments
        for segment in segments:
            for atom in segment:
                atom_counts[atom] += 1
        
        # Include atoms that appear in at least threshold fraction
        min_count = int(threshold * total_segments)
        return {atom for atom, count in atom_counts.items() if count >= min_count}
    
    # Test different thresholds
    for threshold in [0.5, 0.75, 1.0]:
        result = majority_vote(segments, threshold)
        percentage = int(threshold * 100)
        min_segments = int(threshold * len(segments))
        print(f"\nMAJORITY VOTE (threshold={threshold}):")
        print(f"  Requires atom in ≥{min_segments}/{len(segments)} segments ({percentage}%)")
        print(f"  Result: {result}")
        
        if threshold == 0.75:
            print(f"  'Clear' INCLUDED because present in 3/4 segments (75% ≥ 75%)")
    
    print(f"\n=== KEY INSIGHTS ===")
    print(f"• Intersection (threshold=1.0): Very conservative, eliminates atoms with ANY missing occurrence")  
    print(f"• Majority vote (threshold=0.5): Includes atoms present in most segments")
    print(f"• Majority vote (threshold=0.75): Balance between robustness and completeness")
    print(f"• Lower threshold = more preconditions learned, higher chance of including important conditions")
    print(f"• Higher threshold = fewer preconditions learned, more conservative/safe")

if __name__ == "__main__":
    demonstrate_precondition_learning()