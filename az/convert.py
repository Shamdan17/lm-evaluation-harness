#!/usr/bin/env python3
"""
Convert Hugging Face dataset test split to JSONL format
"""

import json
import os
from datasets import load_from_disk

def convert_dataset_to_jsonl(dataset_dir, output_file=None):
    """
    Convert the test split of a Hugging Face dataset to JSONL format
    
    Args:
        dataset_dir (str): Path to the dataset directory
        output_file (str): Output JSONL file path (optional)
    """
    
    # Load the dataset from disk
    print(f"Loading dataset from {dataset_dir}...")
    try:
        dataset = load_from_disk(dataset_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Check if test split exists
    if 'test' not in dataset:
        print("Available splits:", list(dataset.keys()))
        print("Error: 'test' split not found in dataset")
        return
    
    test_dataset = dataset['test']
    print(f"Found test split with {len(test_dataset)} entries")
    
    # Generate output filename if not provided
    if output_file is None:
        dataset_name = os.path.basename(dataset_dir.rstrip('/'))
        output_file = f"{dataset_name}_test.jsonl"
    
    # Convert to JSONL
    print(f"Converting to JSONL format: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(test_dataset):
            # Convert the entry to a JSON string and write to file
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
            
            # Progress indicator for large datasets
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} entries...")
    
    print(f"Successfully converted {len(test_dataset)} entries to {output_file}")
    
    # Show a sample of the first entry
    if len(test_dataset) > 0:
        print("\nSample entry (first record):")
        print(json.dumps(test_dataset[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # Convert the arc_az dataset
    dataset_directory = "arc_az"
    
    # Check if directory exists
    if not os.path.exists(dataset_directory):
        print(f"Error: Directory '{dataset_directory}' not found")
        print("Please make sure the dataset directory exists in the current path")
    else:
        convert_dataset_to_jsonl(dataset_directory)
