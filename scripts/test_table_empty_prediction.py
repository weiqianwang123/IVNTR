"""Test script for TableEmpty prediction using real-world data."""

import os
import subprocess
import sys


def test_table_empty_predictions(max_samples_per_class=None):
    """Test the TableEmpty predictor on all available data."""
    
    data_dir = "Real-World-Data"
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found!")
        return
    
    # Get all available files
    teapot_dir = os.path.join(data_dir, "Teapot")
    table_empty_dir = os.path.join(data_dir, "TableEmpty")
    table_full_dir = os.path.join(data_dir, "TableFull")
    
    # Check directories exist
    for directory in [teapot_dir, table_empty_dir, table_full_dir]:
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            return
    
    # Get all .ply files from each directory
    teapot_files = sorted([f for f in os.listdir(teapot_dir) if f.endswith('.ply')])
    table_empty_files = sorted([f for f in os.listdir(table_empty_dir) if f.endswith('.ply')])
    table_full_files = sorted([f for f in os.listdir(table_full_dir) if f.endswith('.ply')])
    
    print(f"Found {len(teapot_files)} teapot files")
    print(f"Found {len(table_empty_files)} empty table files")
    print(f"Found {len(table_full_files)} full table files")
    
    test_cases = []
    
    # Create positive examples (empty table)
    min_empty_samples = min(len(teapot_files), len(table_empty_files))
    if max_samples_per_class:
        min_empty_samples = min(min_empty_samples, max_samples_per_class)
    
    for i in range(min_empty_samples):
        test_cases.append({
            "name": f"Empty table test {i+1}",
            "teapot": os.path.join(teapot_dir, teapot_files[i]),
            "table": os.path.join(table_empty_dir, table_empty_files[i]),
            "expected": True
        })
    
    # Create negative examples (full table)
    min_full_samples = min(len(teapot_files), len(table_full_files))
    if max_samples_per_class:
        min_full_samples = min(min_full_samples, max_samples_per_class)
    
    for i in range(min_full_samples):
        test_cases.append({
            "name": f"Full table test {i+1}",
            "teapot": os.path.join(teapot_dir, teapot_files[i]),
            "table": os.path.join(table_full_dir, table_full_files[i]),
            "expected": False
        })
    
    print(f"Generated {len(test_cases)} test cases total")
    print(f"  - {min_empty_samples} empty table tests")
    print(f"  - {min_full_samples} full table tests")
    
    print("Testing TableEmpty Prediction")
    print("=" * 50)
    
    # Check if trained model exists
    model_path = "/home/qianwei/IVNTR/table_empty_classifier.pth"
    if os.path.exists(model_path):
        print(f"Using trained model: {model_path}")
        model_arg = f"--model {model_path}"
    else:
        print("Warning: No trained model found. Using random weights.")
        print("For accurate results, train a model first:")
        print("python scripts/train_table_empty_predicate.py --data_dir Real-World-Data")
        model_arg = ""
    
    print()
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        
        # Check if files exist
        if not os.path.exists(test_case['teapot']):
            print(f"  Skipping - teapot file not found: {test_case['teapot']}")
            continue
        if not os.path.exists(test_case['table']):
            print(f"  Skipping - table file not found: {test_case['table']}")
            continue
        
        # Run prediction
        cmd = f"python scripts/predict_table_empty.py {test_case['teapot']} {test_case['table']} {model_arg} --verbose"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            
            # Extract prediction and confidence
            prediction = None
            confidence = None
            
            for line in output_lines:
                if "Table Empty:" in line:
                    prediction = "True" in line
                elif "Confidence:" in line:
                    confidence = float(line.split(":")[1].strip())
            
            if prediction is not None:
                correct = prediction == test_case['expected']
                status = "✓" if correct else "✗"
                
                print(f"  Prediction: {prediction}")
                print(f"  Expected: {test_case['expected']}")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Result: {status} {'Correct' if correct else 'Incorrect'}")
                
                results.append({
                    'test': test_case['name'],
                    'correct': correct,
                    'prediction': prediction,
                    'confidence': confidence
                })
            else:
                print(f"  Error: Could not parse prediction from output")
                print(f"  Output: {result.stdout}")
                print(f"  Error: {result.stderr}")
        
        except Exception as e:
            print(f"  Error running test: {e}")
        
        print()
    
    # Detailed Summary
    if results:
        print("Detailed Results Summary")
        print("=" * 50)
        
        # Overall accuracy
        correct_predictions = sum(1 for r in results if r['correct'])
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"Overall Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2%})")
        
        # Separate accuracy for empty vs full table predictions
        empty_results = [r for r in results if 'Empty table' in r['test']]
        full_results = [r for r in results if 'Full table' in r['test']]
        
        if empty_results:
            empty_correct = sum(1 for r in empty_results if r['correct'])
            empty_total = len(empty_results)
            empty_accuracy = empty_correct / empty_total if empty_total > 0 else 0
            print(f"Empty Table Accuracy: {empty_correct}/{empty_total} ({empty_accuracy:.2%})")
        
        if full_results:
            full_correct = sum(1 for r in full_results if r['correct'])
            full_total = len(full_results)
            full_accuracy = full_correct / full_total if full_total > 0 else 0
            print(f"Full Table Accuracy: {full_correct}/{full_total} ({full_accuracy:.2%})")
        
        # Confidence statistics
        confidences = [r['confidence'] for r in results if r['confidence'] is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            correct_confidences = [r['confidence'] for r in results if r['correct'] and r['confidence'] is not None]
            incorrect_confidences = [r['confidence'] for r in results if not r['correct'] and r['confidence'] is not None]
            
            print(f"\nConfidence Statistics:")
            print(f"Average confidence: {avg_confidence:.4f}")
            if correct_confidences:
                print(f"Average confidence (correct): {sum(correct_confidences)/len(correct_confidences):.4f}")
            if incorrect_confidences:
                print(f"Average confidence (incorrect): {sum(incorrect_confidences)/len(incorrect_confidences):.4f}")
        
        # Show failed cases
        failed_cases = [r for r in results if not r['correct']]
        if failed_cases:
            print(f"\nFailed Cases ({len(failed_cases)}):")            
            for case in failed_cases[:10]:  # Show first 10 failed cases
                print(f"  - {case['test']}: predicted {case['prediction']}, conf {case['confidence']:.4f}")
            if len(failed_cases) > 10:
                print(f"  ... and {len(failed_cases) - 10} more")
        
        if accuracy < 0.6:
            print("\nNote: Low accuracy is expected without a trained model.")
            print("Train a model first for accurate predictions:")
            print("python scripts/train_table_empty_predicate.py --data_dir Real-World-Data")
        elif accuracy > 0.9:
            print("\nExcellent performance! The model is working well.")
        elif accuracy > 0.7:
            print("\nGood performance, but there's room for improvement.")
            print("Consider training longer or adjusting hyperparameters.")


def demo_simple_usage():
    """Demonstrate simple command-line usage."""
    
    print("\nSimple Usage Examples:")
    print("=" * 50)
    
    data_dir = "Real-World-Data"
    
    # Example files
    teapot_file = os.path.join(data_dir, "Teapot", "cloud_1.ply")
    empty_table_file = os.path.join(data_dir, "TableEmpty", "cloud_1.ply")
    full_table_file = os.path.join(data_dir, "TableFull", "cloud_1.ply")
    
    if all(os.path.exists(f) for f in [teapot_file, empty_table_file, full_table_file]):
        
        print("1. Basic usage (True/False output):")
        print(f"python scripts/predict_table_empty.py {teapot_file} {empty_table_file}")
        
        print("\n2. Verbose output:")
        print(f"python scripts/predict_table_empty.py {teapot_file} {empty_table_file} --verbose")
        
        print("\n3. With trained model:")
        print(f"python scripts/predict_table_empty.py {teapot_file} {empty_table_file} --model table_empty_classifier.pth")
        
        print("\n4. Force CPU usage:")
        print(f"python scripts/predict_table_empty.py {teapot_file} {empty_table_file} --device cpu")
        
        print("\nExit codes:")
        print("  0: Table is empty")
        print("  1: Table is not empty") 
        print("  2: Error occurred")
        
    else:
        print("Sample files not found in Real-World-Data directory")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TableEmpty prediction on all data')
    parser.add_argument('--max-samples', type=int, 
                       help='Maximum samples per class (for quick testing)')
    parser.add_argument('--demo-only', action='store_true',
                       help='Only show usage examples, don\'t run tests')
    
    args = parser.parse_args()
    
    if args.demo_only:
        demo_simple_usage()
    else:
        test_table_empty_predictions(max_samples_per_class=args.max_samples)
        if not args.max_samples:  # Only show demo for full test runs
            demo_simple_usage()