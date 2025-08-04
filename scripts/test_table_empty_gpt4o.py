"""Test script for TableEmpty prediction using GPT-4o vision model."""

import os
import base64
import json
import argparse
from typing import Dict, List, Tuple, Optional
import requests
from dotenv import load_dotenv


def load_api_key() -> str:
    """Load OpenAI API key from environment."""
    # Try to load from env.local file
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or .env.local file")
    
    return api_key


def encode_image(image_path: str) -> str:
    """Encode image to base64 for GPT-4o."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def classify_table_empty_gpt4o(teapot_image: str, table_image: str, api_key: str) -> Dict:
    """Use GPT-4o to classify if table is empty given teapot and table images."""
    
    # Encode images
    teapot_b64 = encode_image(teapot_image)
    table_b64 = encode_image(table_image)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You are analyzing two two images to determine if a table has enough space  relative to a reference teapot object.

Task: Compare these two images and determine if the table surface is have enough space for teapot.

Image 1: Reference teapot object
Image 2: Table surface 
Rules:
1 Consider the relative sizes and shapes of objects


Respond with a JSON object containing:
- "not enough": boolean (true if table is have not enough space to place teapot, false otherwise)
- "confidence": float between 0.0 and 1.0
- "explanation": string describing your reasoning
- "objects_detected": list of objects you can identify

Example response:
{
  "not enough": true,
  "confidence": 0.95,
  "explanation": "The table has too much clutter and not enough space for the teapot.",
  "objects_detected": ["teapot", "cup", "plate"]
}"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{teapot_b64}"
                        }
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{table_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", 
                           headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    
    # Parse JSON response
    try:
        parsed_result = json.loads(content)
        return parsed_result
    except json.JSONDecodeError:
        # Fallback parsing if JSON is malformed
        return {
            "empty": "true" in content.lower(),
            "confidence": 0.5,
            "explanation": content,
            "objects_detected": []
        }


def get_rgb_image_path(data_path: str) -> str:
    """Convert a point cloud path to corresponding RGB image path."""
    # Replace .ply with .png and cloud_ with rgb_
    if data_path.endswith('.ply'):
        base_name = os.path.basename(data_path).replace('cloud_', 'rgb_').replace('.ply', '.png')
        return os.path.join(os.path.dirname(data_path), base_name)
    elif data_path.endswith('.png'):
        return data_path
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


def test_single_case(teapot_path: str, table_path: str, api_key: str, 
                    expected: Optional[bool] = None, verbose: bool = True) -> Dict:
    """Test a single table empty classification case."""
    
    try:
        # Get RGB image paths
        teapot_image = get_rgb_image_path(teapot_path)
        table_image = get_rgb_image_path(table_path)
        
        if verbose:
            print(f"Using teapot image: {teapot_image}")
            print(f"Using table image: {table_image}")
        
        # Check if images exist
        if not os.path.exists(teapot_image):
            raise FileNotFoundError(f"Teapot image not found: {teapot_image}")
        if not os.path.exists(table_image):
            raise FileNotFoundError(f"Table image not found: {table_image}")
        
        # Classify using GPT-4o
        if verbose:
            print("Calling GPT-4o for classification...")
        result = classify_table_empty_gpt4o(teapot_image, table_image, api_key)
        
        # Add evaluation metrics if expected result provided
        if expected is not None:
            result['expected'] = expected
            result['correct'] = result['empty'] == expected
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "empty": None,
            "confidence": 0.0,
            "explanation": f"Error occurred: {e}",
            "objects_detected": []
        }


def test_dataset(data_dir: str = "Real-World-Data", max_samples: Optional[int] = None) -> Dict:
    """Test GPT-4o on the full dataset."""
    
    api_key = load_api_key()
    
    # Define directories
    teapot_dir = os.path.join(data_dir, "Teapot")
    table_empty_dir = os.path.join(data_dir, "TableEmpty") 
    table_full_dir = os.path.join(data_dir, "TableFull")
    
    # Check directories exist
    for directory in [teapot_dir, table_empty_dir, table_full_dir]:
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            return {}
    
    # Get files (prefer RGB images, fallback to PLY)
    teapot_files = sorted([f for f in os.listdir(teapot_dir) if f.startswith('rgb_') and f.endswith('.png')])
    if not teapot_files:
        teapot_files = sorted([f for f in os.listdir(teapot_dir) if f.endswith('.ply')])
    
    table_empty_files = sorted([f for f in os.listdir(table_empty_dir) if f.startswith('rgb_') and f.endswith('.png')])
    if not table_empty_files:
        table_empty_files = sorted([f for f in os.listdir(table_empty_dir) if f.endswith('.ply')])
    
    table_full_files = sorted([f for f in os.listdir(table_full_dir) if f.startswith('rgb_') and f.endswith('.png')])
    if not table_full_files:
        table_full_files = sorted([f for f in os.listdir(table_full_dir) if f.endswith('.ply')])
    
    print(f"Found {len(teapot_files)} teapot files")
    print(f"Found {len(table_empty_files)} empty table files")
    print(f"Found {len(table_full_files)} full table files")
    
    results = []
    
    # Test empty tables (expected: True)
    min_empty = min(len(teapot_files), len(table_empty_files))
    if max_samples:
        min_empty = min(min_empty, max_samples)
    
    print(f"\nTesting {min_empty} empty table cases...")
    for i in range(min_empty):
        print(f"\nEmpty table test {i+1}/{min_empty}")
        teapot_path = os.path.join(teapot_dir, teapot_files[i])
        table_path = os.path.join(table_empty_dir, table_empty_files[i])
        
        result = test_single_case(teapot_path, table_path, api_key, expected=True)
        result['test_name'] = f"Empty table {i+1}"
        results.append(result)
        
        # Show result
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            status = "✅" if result.get('correct', False) else "❌"
            print(f"{status} Predicted: {result['empty']}, Expected: True")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Objects: {result['objects_detected']}")
    
    # Test full tables (expected: False)
    min_full = min(len(teapot_files), len(table_full_files))
    if max_samples:
        min_full = min(min_full, max_samples)
    
    print(f"\nTesting {min_full} full table cases...")
    for i in range(min_full):
        print(f"\nFull table test {i+1}/{min_full}")
        teapot_path = os.path.join(teapot_dir, teapot_files[i])
        table_path = os.path.join(table_full_dir, table_full_files[i])
        
        result = test_single_case(teapot_path, table_path, api_key, expected=False)
        result['test_name'] = f"Full table {i+1}"
        results.append(result)
        
        # Show result
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            status = "✅" if result.get('correct', False) else "❌"
            print(f"{status} Predicted: {result['empty']}, Expected: False")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Objects: {result['objects_detected']}")
    
    return {"results": results}


def print_summary(test_results: Dict):
    """Print test summary statistics."""
    
    results = test_results.get("results", [])
    if not results:
        print("No results to summarize")
        return
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Filter out error cases
    valid_results = [r for r in results if 'error' not in r]
    error_count = len(results) - len(valid_results)
    
    if error_count > 0:
        print(f"⚠️  {error_count} tests failed with errors")
    
    if not valid_results:
        print("No valid results to analyze")
        return
    
    # Overall accuracy
    correct_predictions = sum(1 for r in valid_results if r.get('correct', False))
    total_predictions = len(valid_results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"Overall Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
    
    # Breakdown by test type
    empty_results = [r for r in valid_results if 'Empty table' in r['test_name']]
    full_results = [r for r in valid_results if 'Full table' in r['test_name']]
    
    if empty_results:
        empty_correct = sum(1 for r in empty_results if r.get('correct', False))
        empty_accuracy = empty_correct / len(empty_results)
        print(f"Empty Table Accuracy: {empty_correct}/{len(empty_results)} ({empty_accuracy:.1%})")
    
    if full_results:
        full_correct = sum(1 for r in full_results if r.get('correct', False))
        full_accuracy = full_correct / len(full_results)
        print(f"Full Table Accuracy: {full_correct}/{len(full_results)} ({full_accuracy:.1%})")
    
    # Confidence statistics
    confidences = [r['confidence'] for r in valid_results]
    avg_confidence = sum(confidences) / len(confidences)
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    # Show some example explanations
    print(f"\nExample GPT-4o Explanations:")
    for i, result in enumerate(valid_results[:3]):
        print(f"{i+1}. {result['test_name']}: {result['explanation'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description='Test table empty classification using GPT-4o')
    parser.add_argument('--data-dir', default='Real-World-Data',
                       help='Directory containing test data')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum samples per class for testing')
    parser.add_argument('--teapot', type=str,
                       help='Single teapot file to test')
    parser.add_argument('--table', type=str, 
                       help='Single table file to test')
    parser.add_argument('--expected', type=bool,
                       help='Expected result for single test')
    
    args = parser.parse_args()
    
    try:
        if args.teapot and args.table:
            # Single test case
            api_key = load_api_key()
            print(f"Testing single case:")
            print(f"Teapot: {args.teapot}")
            print(f"Table: {args.table}")
            
            result = test_single_case(args.teapot, args.table, api_key, 
                                    expected=args.expected, verbose=True)
            
            print(f"\nResult:")
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"Empty: {result['empty']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Objects detected: {result['objects_detected']}")
                print(f"Explanation: {result['explanation']}")
                
                if args.expected is not None:
                    status = "✅ Correct" if result['correct'] else "❌ Incorrect"
                    print(f"Result: {status} (expected {args.expected})")
        else:
            # Full dataset test
            print("Testing GPT-4o on table empty classification dataset")
            test_results = test_dataset(args.data_dir, args.max_samples)
            print_summary(test_results)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())