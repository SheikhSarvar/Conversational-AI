# TechBot: Industry-Specific Conversational AI for IT Support

## Overview

**TechBot** is an advanced conversational AI specialized for the Technology and Information Technology (IT) sector. This project leverages the **Llama-3.2-1B-Instruct** Large Language Model (LLM), fine-tuned using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA (Low-Rank Adaptation)** on curated IT-specific datasets.

The goal is to provide a highly accurate, context-aware technical assistant capable of handling software troubleshooting, programming queries, and infrastructure support.

## Key Features

- **Domain Specialization**: Fine-tuned on technical documentation, programming forums, and IT support logs.
- **Efficient Inference**: Supports 8-bit quantization for GPU-efficient deployment.
- **Context-Aware**: Uses advanced chat templates for multi-turn technical dialogues.
- **Lightweight Architecture**: Based on Llama-3.2-1B, providing a balance between performance and resource efficiency.

## Project Structure

```
Conversational-AI/
├── config/                 # Training and model configurations
├── data/                   # Data pipeline (raw, processed, validation)
├── models/                 # Fine-tuned LoRA adapters and checkpoints
├── notebooks/              # Research, fine-tuning, and evaluation notebooks
├── scripts/                # Data collection and preprocessing utilities
├── src/                    # Core application logic (chatbot, model handler)
├── tests/                  # Test queries and validation suites
├── PROJECT_DOCS.md         # Comprehensive project documentation (Research Focus)
└── README.md               # Quick start and overview
```

## Technology Stack

- **Base Model**: Llama-3.2-1B-Instruct
- **Fine-Tuning**: LoRA (via PEFT)
- **Frameworks**: Hugging Face Transformers, PyTorch
- **Data Processing**: Pandas, NumPy
- **Interface**: CLI (TechBot CLI)
- **Quantization**: BitsAndBytes (8-bit)

## Quick Start

### 1. Install Dependencies

Ensure you have [uv](https://github.com/astral-sh/uv) installed:

```bash
uv sync
```

### 2. Run TechBot

You can run the bot in different modes depending on your hardware:

**CPU Mode:**
```bash
uv run python src/chatbot.py --adapter models/conversational-finetuned
```

**GPU Mode (8-bit Quantized):**
```bash
uv run python src/chatbot.py --adapter models/conversational-finetuned --8bit
```

## Project Documentation

For a detailed breakdown of the research methodology, architecture, and evaluation results, please refer to:
👉 **[PROJECT_DOCS.md](./PROJECT_DOCS.md)**

## Research Paper

The findings of this project are documented in a comprehensive research paper located in:
`research_paper/main_paper.md`

## Author

[Your Name]
Woolf University
Master's in CS: Artificial Intelligence and Machine Learning
