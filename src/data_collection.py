"""
Data Collection Script for IT Support Conversations
This script collects IT support data from various sources including
forums, documentation, and public datasets.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import json
import os

class ITSupportDataCollector:
    """Collects IT support data from multiple sources."""
    
    def __init__(self):
        self.data = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_stackoverflow(self, tags: List[str], max_questions: int = 1000) -> List[Dict]:
        """
        Scrape IT support questions from Stack Overflow.
        
        Args:
            tags: List of tags to filter questions (e.g., ['windows', 'networking'])
            max_questions: Maximum number of questions to collect
            
        Returns:
            List of dictionaries containing question-answer pairs
        """
        base_url = "https://api.stackexchange.com/2.3/questions"
        collected_data = []
        
        for tag in tags:
            params = {
                'order': 'desc',
                'sort': 'votes',
                'tagged': tag,
                'site': 'stackoverflow',
                'filter': 'withbody',
                'pagesize': 100
            }
            
            try:
                response = requests.get(base_url, params=params, headers=self.headers)
                response.raise_for_status()
                questions = response.json().get('items', [])
                
                for q in questions[:max_questions]:
                    if q.get('accepted_answer_id'):
                        # Fetch the accepted answer
                        answer_url = f"https://api.stackexchange.com/2.3/answers/{q['accepted_answer_id']}"
                        answer_params = {'site': 'stackoverflow', 'filter': 'withbody'}
                        answer_response = requests.get(answer_url, params=answer_params)
                        
                        if answer_response.status_code == 200:
                            answer_data = answer_response.json().get('items', [])
                            if answer_data:
                                collected_data.append({
                                    'question': q.get('title'),
                                    'question_body': q.get('body'),
                                    'response': answer_data[0].get('body'),
                                    'tags': ','.join(q.get('tags', [])),
                                    'score': q.get('score'),
                                    'source': 'stackoverflow'
                                })
                        
                        time.sleep(0.5)  # Rate limiting
                
                time.sleep(1)  # Rate limiting between tags
                
            except Exception as e:
                print(f"Error scraping Stack Overflow for tag {tag}: {e}")
        
        return collected_data
    
    def generate_synthetic_data(self, num_samples: int = 100) -> List[Dict]:
        """
        Generate synthetic IT support Q&A pairs for common scenarios.
        
        Args:
            num_samples: Number of synthetic examples to generate
            
        Returns:
            List of synthetic Q&A pairs
        """
        synthetic_data = []
        
        # Common IT support scenarios
        scenarios = [
            {
                'question': 'How do I reset my Windows password?',
                'response': 'To reset your Windows password, follow these steps:\n1. Boot into Safe Mode by pressing F8 during startup\n2. Log in as Administrator\n3. Go to Control Panel > User Accounts\n4. Select the account you want to reset\n5. Click "Change password" and enter a new password\n6. Restart your computer normally\n\nAlternatively, you can use a password reset disk if you created one previously.'
            },
            {
                'question': 'My computer is running very slow. What should I do?',
                'response': 'Here are several steps to improve your computer\'s performance:\n1. Check Task Manager (Ctrl+Shift+Esc) to identify resource-heavy processes\n2. Uninstall unnecessary programs from Control Panel\n3. Run Disk Cleanup to remove temporary files\n4. Disable startup programs that aren\'t needed\n5. Check for malware using Windows Defender or antivirus software\n6. Ensure you have at least 15% free disk space\n7. Consider upgrading RAM if you frequently run multiple applications\n8. Update your drivers and Windows\n\nIf the problem persists, consider checking for hardware issues or performing a clean Windows installation.'
            },
            {
                'question': 'How do I fix "No Internet Connection" error?',
                'response': 'Try these troubleshooting steps:\n1. Restart your router and modem (unplug for 30 seconds)\n2. Check if other devices can connect to the network\n3. Run Windows Network Troubleshooter (right-click network icon)\n4. Update network adapter drivers\n5. Reset TCP/IP stack: Open Command Prompt as admin and run:\n   - netsh winsock reset\n   - netsh int ip reset\n   - ipconfig /release\n   - ipconfig /renew\n   - ipconfig /flushdns\n6. Disable and re-enable your network adapter\n7. Check if your IP address is valid (should not be 169.254.x.x)\n8. Temporarily disable firewall/antivirus to test\n\nIf using Wi-Fi, try connecting via Ethernet to isolate the issue.'
            },
            {
                'question': 'How do I install a printer on Windows 10?',
                'response': 'To install a printer on Windows 10:\n1. Connect the printer to your computer via USB or ensure it\'s on the same network\n2. Go to Settings > Devices > Printers & scanners\n3. Click "Add a printer or scanner"\n4. Windows will search for available printers\n5. Select your printer from the list and click "Add device"\n6. If not found, click "The printer that I want isn\'t listed"\n7. Choose the appropriate option (local printer, network printer, etc.)\n8. Follow the wizard to install drivers\n\nFor network printers, you may need the IP address. You can also download drivers from the manufacturer\'s website for best compatibility.'
            },
            {
                'question': 'What should I do if my laptop won\'t turn on?',
                'response': 'Try these steps to diagnose the issue:\n1. Check if the power adapter is working (LED light on adapter)\n2. Remove the battery and hold power button for 30 seconds\n3. Reconnect power adapter (without battery) and try to turn on\n4. If it works, the battery may be faulty\n5. Check for signs of physical damage\n6. Listen for beep codes or fan sounds when pressing power\n7. Try connecting to an external monitor to rule out display issues\n8. Remove all peripherals and try again\n9. If you see lights but no display, try pressing Fn+F4/F5 to switch displays\n\nIf none of these work, there may be a hardware failure requiring professional repair.'
            }
        ]
        
        # Add variations and more scenarios
        for scenario in scenarios[:num_samples]:
            synthetic_data.append({
                'question': scenario['question'],
                'response': scenario['response'],
                'source': 'synthetic',
                'tags': 'it-support,troubleshooting',
                'score': 0
            })
        
        return synthetic_data
    
    def load_ubuntu_dialogue_corpus(self, file_path: str) -> List[Dict]:
        """
        Load and process Ubuntu Dialogue Corpus.
        
        Args:
            file_path: Path to the Ubuntu Dialogue Corpus file
            
        Returns:
            List of processed dialogue pairs
        """
        dialogues = []
        
        if not os.path.exists(file_path):
            print(f"Ubuntu corpus file not found: {file_path}")
            return dialogues
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    dialogue = json.loads(line)
                    dialogues.append({
                        'question': dialogue.get('context', ''),
                        'response': dialogue.get('response', ''),
                        'source': 'ubuntu_corpus',
                        'tags': 'linux,ubuntu',
                        'score': 0
                    })
        except Exception as e:
            print(f"Error loading Ubuntu corpus: {e}")
        
        return dialogues
    
    def save_data(self, output_path: str):
        """Save collected data to CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(self.data)} records to {output_path}")
        print(f"Data sources: {df['source'].value_counts().to_dict()}")

# Usage example
if __name__ == "__main__":
    collector = ITSupportDataCollector()
    
    print("Starting data collection...")
    
    # Generate synthetic data (immediate, no API required)
    print("\n1. Generating synthetic IT support data...")
    synthetic_data = collector.generate_synthetic_data(num_samples=100)
    collector.data.extend(synthetic_data)
    print(f"Generated {len(synthetic_data)} synthetic examples")
    
    # Collect from Stack Overflow (requires API access)
    print("\n2. Collecting from Stack Overflow...")
    it_tags = ['windows', 'linux', 'networking', 'troubleshooting', 'hardware']
    try:
        stackoverflow_data = collector.scrape_stackoverflow(it_tags, max_questions=50)
        collector.data.extend(stackoverflow_data)
        print(f"Collected {len(stackoverflow_data)} examples from Stack Overflow")
    except Exception as e:
        print(f"Could not collect from Stack Overflow: {e}")
        print("Continuing with synthetic data only...")
    
    # Save collected data
    print("\n3. Saving collected data...")
    collector.save_data('data/raw/it_support_data.csv')
    
    print("\nData collection complete!")
