"""
Data Validation Script

Validates the quality and format of collected and processed data.

Usage:
    python scripts/validate_data.py --data-dir data/processed
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def validate_dataset(file_path: Path, format_type: str = "chat") -> Dict[str, Any]:
    """Validate a single dataset file"""
    
    print(f"\n🔍 Validating {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        "file": file_path.name,
        "total_samples": len(data),
        "valid_samples": 0,
        "invalid_samples": 0,
        "errors": []
    }
    
    for idx, entry in enumerate(data):
        try:
            if format_type == "chat":
                # Validate chat format
                assert "messages" in entry, "Missing 'messages' key"
                assert isinstance(entry["messages"], list), "'messages' must be a list"
                assert len(entry["messages"]) >= 2, "Need at least 2 messages"
                
                for msg in entry["messages"]:
                    assert "role" in msg, "Message missing 'role'"
                    assert "content" in msg, "Message missing 'content'"
                    assert msg["role"] in ["system", "user", "assistant"], f"Invalid role: {msg['role']}"
            else:
                # Validate instruction format
                assert "instruction" in entry, "Missing 'instruction' key"
                assert "input" in entry, "Missing 'input' key"
                assert "output" in entry, "Missing 'output' key"
            
            stats["valid_samples"] += 1
            
        except AssertionError as e:
            stats["invalid_samples"] += 1
            stats["errors"].append(f"Sample {idx}: {str(e)}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory to validate"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["chat", "instruction"],
        default="chat",
        help="Expected data format"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Find all dataset files
    dataset_files = []
    for split in ["train", "validation", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            dataset_files.extend(list(split_dir.glob("*.json")))
    
    if not dataset_files:
        print(f"❌ No dataset files found in {data_dir}")
        return
    
    print(f"📁 Found {len(dataset_files)} dataset files")
    
    all_stats = []
    for file_path in dataset_files:
        stats = validate_dataset(file_path, format_type=args.format)
        all_stats.append(stats)
        
        print(f"  ✅ Valid: {stats['valid_samples']}")
        print(f"  ❌ Invalid: {stats['invalid_samples']}")
        
        if stats['errors']:
            print(f"  ⚠️  First 5 errors:")
            for error in stats['errors'][:5]:
                print(f"     - {error}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    total_samples = sum(s['total_samples'] for s in all_stats)
    total_valid = sum(s['valid_samples'] for s in all_stats)
    total_invalid = sum(s['invalid_samples'] for s in all_stats)
    
    print(f"Total samples: {total_samples}")
    print(f"Valid samples: {total_valid} ({total_valid/total_samples*100:.2f}%)")
    print(f"Invalid samples: {total_invalid} ({total_invalid/total_samples*100:.2f}%)")
    
    if total_invalid == 0:
        print("\n✅ All datasets are valid!")
    else:
        print(f"\n⚠️  Found {total_invalid} invalid samples. Please review and fix.")


if __name__ == "__main__":
    main()
