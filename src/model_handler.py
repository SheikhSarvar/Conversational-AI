"""
Model Handler for TechBot
Handles model loading, inference, and response generation.
Supports CPU, GPU, and optional 8-bit quantization.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from typing import Optional, List, Dict
import os
import warnings
warnings.filterwarnings('ignore')


class ModelHandler:
    """Handles model loading and inference for TechBot"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_8bit: bool = False,
    ):
        print(f"🤖 Loading base model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.has_gpu = self.device == "cuda"

        # ------------------------------
        # Quantization config for 8-bit GPU
        # ------------------------------
        quantization_config = None
        if load_in_8bit and self.has_gpu:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )

        # ------------------------------
        # Load tokenizer
        # ------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------------------------------
        # Load base model
        # ------------------------------
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.has_gpu else None,
            torch_dtype=torch.float16 if self.has_gpu else torch.float32,
            quantization_config=quantization_config,
            trust_remote_code=True
        )

        # ------------------------------
        # Load LoRA adapter
        # ------------------------------
        if adapter_path:
            print(f"🔧 Loading LoRA adapter: {adapter_path}")
            kwargs = {"device_map": "auto"} if self.has_gpu else {}
            if not self.has_gpu:
                kwargs["device_map"] = None
                kwargs["offload_folder"] = None  # CPU-only, no offload

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                **kwargs
            )

            # Merge LoRA if FP16 (only safe for GPU)
            if not load_in_8bit and self.has_gpu:
                print("🔗 Merging LoRA weights into base model...")
                self.model = self.model.merge_and_unload()
            else:
                print("⚠️ Skipping merge: CPU or 8-bit mode")

        # ------------------------------
        # Pipeline for text generation
        # ------------------------------
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if self.has_gpu else None
        )

        print("✅ Model loaded successfully on", self.device.upper())

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )

        return outputs[0]['generated_text'].strip()
