#!/usr/bin/env python3
"""
Complete empty JSON files in labels_1000 folder using data from data_train.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


def load_train_data(train_file: str) -> Dict[str, Any]:
    """Load data_train.json and index by img_id."""
    print(f"Loading training data from: {train_file}")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Index by img_id for fast lookup
    img_id_map = {}
    for entry in train_data:
        if isinstance(entry, dict) and 'img_id' in entry:
            img_id = str(entry['img_id'])
            img_id_map[img_id] = entry
    
    print(f"Loaded {len(img_id_map)} entries from training data")
    return img_id_map


def find_empty_json_files(labels_dir: str) -> List[str]:
    """Find all JSON files that are empty (contain only [])."""
    empty_files = []
    
    json_files = list(Path(labels_dir).glob("*.json"))
    print(f"\nScanning {len(json_files)} JSON files in {labels_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Check if it's an empty list
            if isinstance(content, list) and len(content) == 0:
                empty_files.append(str(json_file))
        except Exception as e:
            print(f"  ✗ Error reading {json_file.name}: {e}")
    
    return empty_files


def complete_empty_files(empty_files: List[str], img_id_map: Dict[str, Any], dry_run: bool = False):
    """Fill empty JSON files with data from training set."""
    filled_count = 0
    not_found_count = 0
    
    print(f"\n{'DRY RUN: ' if dry_run else ''}Processing {len(empty_files)} empty files...")
    
    for file_path in empty_files:
        # Extract img_id from filename (e.g., "50.json" -> "50")
        filename = Path(file_path).stem
        img_id = filename
        
        # Look up data in training set
        if img_id in img_id_map:
            data = img_id_map[img_id]
            
            if dry_run:
                print(f"  Would fill {filename}.json with img_id={img_id}")
            else:
                # Write the data to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"  ✓ Filled {filename}.json with data for img_id={img_id}")
            
            filled_count += 1
        else:
            print(f"  ✗ No data found for img_id={img_id} ({filename}.json)")
            not_found_count += 1
    
    return filled_count, not_found_count


def unwrap_list_files(labels_dir: str) -> int:
    """Unwrap JSON files that are lists containing dict(s) - take first entry."""
    unwrapped_count = 0
    
    print(f"\nUnwrapping list-wrapped JSON files in {labels_dir}...")
    
    for json_file in Path(labels_dir).glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # If it's a list containing dict(s), unwrap it
            if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
                # Take the first caption/entry
                unwrapped = content[0]
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(unwrapped, f, indent=4, ensure_ascii=False)
                
                print(f"  ✓ Unwrapped {json_file.name} (was list with {len(content)} entries)")
                unwrapped_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
    
    return unwrapped_count


def main():
    # Configuration
    work_dir = os.getcwd()
    labels_dir = os.path.join(work_dir, "labels_1000")
    train_file = os.path.join(work_dir, "data_train.json")
    validation_file = os.path.join(work_dir, "data_validation.json")
    
    print("=" * 70)
    print("Complete Empty JSON Files")
    print("=" * 70)
    print(f"Working directory: {work_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Training data: {train_file}")
    print(f"Validation data: {validation_file}")
    
    # Check if files exist
    if not os.path.exists(labels_dir):
        print(f"\n✗ Error: Labels directory not found: {labels_dir}")
        return 1
    
    if not os.path.exists(train_file):
        print(f"\n✗ Error: Training data file not found: {train_file}")
        return 1
    
    # Load training data
    try:
        img_id_map = load_train_data(train_file)
    except Exception as e:
        print(f"\n✗ Error loading training data: {e}")
        return 1
    
    # Load validation data if available
    if os.path.exists(validation_file):
        try:
            print(f"\nLoading validation data from: {validation_file}")
            with open(validation_file, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
            
            # Merge validation data into img_id_map
            for entry in validation_data:
                if isinstance(entry, dict) and 'img_id' in entry:
                    img_id = str(entry['img_id'])
                    if img_id not in img_id_map:
                        img_id_map[img_id] = entry
            
            print(f"Merged validation data. Total entries: {len(img_id_map)}")
        except Exception as e:
            print(f"⚠ Warning: Could not load validation data: {e}")
    else:
        print(f"\n⚠ Validation data file not found: {validation_file}")
    
    # Load test data if available
    test_file = os.path.join(work_dir, "data_test.json")
    if os.path.exists(test_file):
        try:
            print(f"\nLoading test data from: {test_file}")
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # Merge test data into img_id_map
            for entry in test_data:
                if isinstance(entry, dict) and 'img_id' in entry:
                    img_id = str(entry['img_id'])
                    if img_id not in img_id_map:
                        img_id_map[img_id] = entry
            
            print(f"Merged test data. Total entries: {len(img_id_map)}")
        except Exception as e:
            print(f"⚠ Warning: Could not load test data: {e}")
    
    # Find empty files
    empty_files = find_empty_json_files(labels_dir)
    
    if not empty_files:
        print("\n✓ No empty JSON files found!")
        return 0
    
    print(f"\nFound {len(empty_files)} empty JSON files")
    
    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("Do you want to fill these files? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return 0
    
    # Fill the empty files
    filled_count, not_found_count = complete_empty_files(empty_files, img_id_map, dry_run=False)
    
    # Unwrap list-wrapped files
    unwrapped_count = unwrap_list_files(labels_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total empty files found: {len(empty_files)}")
    print(f"Successfully filled: {filled_count}")
    print(f"Not found in training data: {not_found_count}")
    print(f"List-wrapped files unwrapped: {unwrapped_count}")
    print("\n✓ Operation complete!")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
