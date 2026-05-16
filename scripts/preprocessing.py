"""
Data Preprocessing Script for Technology/IT Industry LLM Bot

This script processes raw collected data and prepares it for model training:
1. Cleans and normalizes text
2. Removes duplicates and low-quality entries
3. Formats data for instruction fine-tuning
4. Splits into train/validation/test sets

Usage:
    python scripts/data_preprocessing.py --input data/raw --output data/processed
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataPreprocessor:
    """Preprocesses collected data for LLM training"""
    
    def __init__(self, min_length: int = 20, max_length: int = 2048):
        self.min_length = min_length
        self.max_length = max_length
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (optional - you might want to keep some)
        # text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters but keep code-related ones
        # text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]\{\}\<\>\=\+\*\/\#\@\!\?]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def is_valid_entry(self, entry: Dict[str, Any]) -> bool:
        """Check if entry meets quality criteria"""
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        
        # Check minimum length
        if len(question) < self.min_length or len(answer) < self.min_length:
            return False
        
        # Check maximum length
        if len(question) > self.max_length or len(answer) > self.max_length:
            return False
        
        # Check for empty or None values
        if not question or not answer:
            return False
        
        # Check for minimum word count
        if len(question.split()) < 3 or len(answer.split()) < 5:
            return False
        
        return True
    
    def format_for_training(self, entry: Dict[str, Any]) -> Dict[str, str]:
        """
        Format entry for instruction fine-tuning
        
        Returns a dictionary with 'instruction', 'input', and 'output' keys
        """
        question = self.clean_text(entry.get("question", ""))
        context = self.clean_text(entry.get("context", ""))
        answer = self.clean_text(entry.get("answer", ""))
        tags = entry.get("tags", [])
        
        # Create instruction based on tags
        if tags:
            tag_str = ", ".join(tags[:3])  # Limit to 3 tags
            instruction = f"You are a technical expert in {tag_str}. Answer the following question accurately and concisely."
        else:
            instruction = "You are a technical expert. Answer the following question accurately and concisely."
        
        # Combine question and context if available
        if context:
            input_text = f"{question}\n\nContext: {context}"
        else:
            input_text = question
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": answer
        }
    
    def format_for_chat(self, entry: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """
        Format entry for chat-based fine-tuning (ChatML format)
        
        Returns a dictionary with 'messages' key containing conversation
        """
        question = self.clean_text(entry.get("question", ""))
        context = self.clean_text(entry.get("context", ""))
        answer = self.clean_text(entry.get("answer", ""))
        
        messages = [
            {
                "role": "system",
                "content": "You are TechBot, an AI assistant specialized in technology and IT. You provide accurate, helpful, and concise answers to technical questions."
            },
            {
                "role": "user",
                "content": f"{question}\n\n{context}" if context else question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
        
        return {"messages": messages}
    
    def process_dataset(
        self,
        input_files: List[Path],
        output_dir: Path,
        format_type: str = "chat"
    ):
        """
        Process all input files and create train/val/test splits
        
        Args:
            input_files: List of input JSON files
            output_dir: Output directory for processed data
            format_type: 'chat' or 'instruction' format
        """
        print("🔄 Processing datasets...")
        
        all_data = []
        
        # Load all data
        for file_path in input_files:
            print(f"📂 Loading {file_path.name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        
        print(f"📊 Total raw entries: {len(all_data)}")
        
        # Filter valid entries
        print("🧹 Filtering and cleaning data...")
        valid_data = []
        
        for entry in tqdm(all_data, desc="Processing"):
            if self.is_valid_entry(entry):
                if format_type == "chat":
                    formatted = self.format_for_chat(entry)
                else:
                    formatted = self.format_for_training(entry)
                
                valid_data.append(formatted)
        
        print(f"✅ Valid entries after filtering: {len(valid_data)}")
        
        # Remove duplicates based on question
        print("🔍 Removing duplicates...")
        seen_questions = set()
        unique_data = []
        
        for entry in valid_data:
            if format_type == "chat":
                question = entry["messages"][1]["content"]
            else:
                question = entry["input"]
            
            # Create a simple hash of the question
            q_hash = hash(question.lower()[:100])
            
            if q_hash not in seen_questions:
                seen_questions.add(q_hash)
                unique_data.append(entry)
        
        print(f"✅ Unique entries: {len(unique_data)}")
        
        # Split into train/val/test
        print("📊 Splitting into train/validation/test sets...")
        
        # 80% train, 10% validation, 10% test
        train_data, temp_data = train_test_split(
            unique_data,
            test_size=0.2,
            random_state=42
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            random_state=42
        )
        
        print(f"📈 Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Save datasets
        output_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = {
            "train": (train_data, output_dir.parent / "train"),
            "validation": (val_data, output_dir.parent / "validation"),
            "test": (test_data, output_dir.parent / "test")
        }
        
        for split_name, (data, split_dir) in datasets.items():
            split_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = split_dir / f"tech_qa_{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {split_name} set: {output_file}")
        
        # Save statistics
        self._save_statistics(train_data, val_data, test_data, output_dir)
        
        print("\n🎉 Data preprocessing complete!")
    
    def _save_statistics(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
        output_dir: Path
    ):
        """Save dataset statistics"""
        stats = {
            "total_samples": len(train_data) + len(val_data) + len(test_data),
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "test_samples": len(test_data),
            "train_percentage": round(len(train_data) / (len(train_data) + len(val_data) + len(test_data)) * 100, 2),
            "val_percentage": round(len(val_data) / (len(train_data) + len(val_data) + len(test_data)) * 100, 2),
            "test_percentage": round(len(test_data) / (len(train_data) + len(val_data) + len(test_data)) * 100, 2),
        }
        
        stats_file = output_dir / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess collected data for LLM training")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input directory containing raw data files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["chat", "instruction"],
        default="chat",
        help="Output format: 'chat' for ChatML or 'instruction' for instruction tuning"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum text length for filtering"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum text length for filtering"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Find all JSON files in input directory
    input_files = list(input_dir.glob("*.json"))
    
    if not input_files:
        print(f"❌ No JSON files found in {input_dir}")
        return
    
    print(f"📁 Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f.name}")
    
    # Process data
    preprocessor = DataPreprocessor(
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    preprocessor.process_dataset(
        input_files=input_files,
        output_dir=output_dir,
        format_type=args.format
    )


if __name__ == "__main__":
    main()
