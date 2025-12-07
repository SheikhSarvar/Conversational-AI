"""
Data Preprocessing Pipeline for IT Support Conversations
Cleans, formats, and prepares data for model training.
"""

import pandas as pd
import re
from typing import List, Dict
import html
import os
from transformers import AutoTokenizer
import json

class ITSupportPreprocessor:
    """Preprocesses IT support data for model training."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        print(f"Initializing preprocessor with tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove code blocks (preserve technical content but remove formatting)
        text = re.sub(r'```[\s\S]*?```', '[CODE BLOCK]', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (but keep technical terms)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Normalize line breaks
        text = text.replace('\\n', ' ').replace('\\r', ' ')
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data based on quality criteria.
        
        Args:
            df: DataFrame with 'question' and 'response' columns
            
        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)
        
        # Ensure required columns exist
        if 'question' not in df.columns or 'response' not in df.columns:
            print("Warning: Missing required columns. Creating from available data...")
            if 'question_body' in df.columns:
                df['question'] = df['question'].fillna('') + ' ' + df['question_body'].fillna('')
        
        # Remove rows with missing data
        df = df.dropna(subset=['question', 'response'])
        
        # Remove empty or very short entries
        df = df[df['question'].str.len() >= 20]
        df = df[df['response'].str.len() >= 50]
        
        # Remove very long entries (likely spam or irrelevant)
        df = df[df['response'].str.len() <= 2000]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['question'], keep='first')
        
        # Remove low-quality indicators
        df = df[~df['response'].str.lower().str.contains('i don\'t know|not sure|can\'t help', na=False)]
        
        filtered_count = len(df)
        print(f"Filtered {initial_count - filtered_count} low-quality entries ({initial_count} -> {filtered_count})")
        
        return df
    
    def format_for_training(self, df: pd.DataFrame) -> List[Dict]:
        """
        Format data for conversational model training.
        
        Args:
            df: DataFrame with 'question' and 'response' columns
            
        Returns:
            List of formatted training examples
        """
        formatted_data = []
        
        for idx, row in df.iterrows():
            # Create conversational format
            conversation = f"User: {row['question']}\nAssistant: {row['response']}"
            
            # Tokenize to check length
            tokens = self.tokenizer.encode(conversation, truncation=True, max_length=512)
            
            if len(tokens) > 50:  # Ensure minimum length
                formatted_data.append({
                    'text': conversation,
                    'length': len(tokens),
                    'source': row.get('source', 'unknown'),
                    'tags': row.get('tags', '')
                })
        
        return formatted_data
    
    def create_train_val_split(self, data: List[Dict], val_ratio: float = 0.1) -> tuple:
        """
        Split data into training and validation sets.
        
        Args:
            data: List of formatted examples
            val_ratio: Ratio of validation data
            
        Returns:
            Tuple of (train_data, val_data)
        """
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * (1 - val_ratio))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        return train_data, val_data
    
    def process_dataset(self, input_path: str, output_dir: str):
        """
        Complete preprocessing pipeline.
        
        Args:
            input_path: Path to raw data CSV
            output_dir: Directory to save processed data
        """
        print(f"\n{'='*60}")
        print("IT Support Data Preprocessing Pipeline")
        print(f"{'='*60}\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        print(f"1. Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        print(f"   Loaded {len(df)} raw examples")
        
        # Clean text
        print("\n2. Cleaning text...")
        df['question'] = df['question'].apply(self.clean_text)
        df['response'] = df['response'].apply(self.clean_text)
        
        # Filter quality
        print("\n3. Filtering quality...")
        df = self.filter_quality(df)
        
        # Format for training
        print("\n4. Formatting for training...")
        formatted_data = self.format_for_training(df)
        print(f"   Created {len(formatted_data)} formatted examples")
        
        # Create train/val split
        print("\n5. Creating train/validation split...")
        train_data, val_data = self.create_train_val_split(formatted_data, val_ratio=0.1)
        print(f"   Training: {len(train_data)} examples")
        print(f"   Validation: {len(val_data)} examples")
        
        # Save processed data
        print("\n6. Saving processed data...")
        train_path = os.path.join(output_dir, 'train.jsonl')
        val_path = os.path.join(output_dir, 'validation.jsonl')
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
        
        # Save statistics
        stats = {
            'total_examples': len(formatted_data),
            'train_examples': len(train_data),
            'val_examples': len(val_data),
            'avg_length_tokens': sum(item['length'] for item in formatted_data) / len(formatted_data),
            'min_length': min(item['length'] for item in formatted_data),
            'max_length': max(item['length'] for item in formatted_data),
            'sources': {}
        }
        
        for item in formatted_data:
            source = item.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Preprocessing Complete!")
        print(f"{'='*60}")
        print(f"\nStatistics:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"  Average length: {stats['avg_length_tokens']:.1f} tokens")
        print(f"  Length range: {stats['min_length']} - {stats['max_length']} tokens")
        print(f"\nData sources:")
        for source, count in stats['sources'].items():
            print(f"  {source}: {count} examples")
        print(f"\nFiles saved:")
        print(f"  Training: {train_path}")
        print(f"  Validation: {val_path}")
        print(f"  Statistics: {stats_path}")

# Usage example
if __name__ == "__main__":
    preprocessor = ITSupportPreprocessor(model_name="microsoft/DialoGPT-medium")
    preprocessor.process_dataset(
        input_path='data/raw/it_support_data.csv',
        output_dir='data/processed'
    )
