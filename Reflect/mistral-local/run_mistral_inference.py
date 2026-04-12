#!/usr/bin/env python3
"""
Run Mistral-7B-Instruct-v0.3 inference on SafeRLHF dataset
Optimized for Apple Silicon (MPS) with checkpoint support

Output format: Nested list structure matching conversation format
[
    [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]],
    [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]],
    ...
]
"""

import json
import os
from pathlib import Path
from typing import List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class MistralInferenceRunner:
    """Handle Mistral model inference with checkpointing and error recovery"""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "mps",
        temperature: float = 0.7,
        max_new_tokens: int = 1500,
        system_prompt_path: str = "data/HHH_ICL.txt",
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        print(f"Initializing Mistral Inference Runner...")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Temperature: {temperature}")
        print(f"Max tokens: {max_new_tokens}")

        # Load system prompt
        print(f"\nLoading system prompt from {system_prompt_path}...")
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()
        print(f"System prompt loaded ({len(self.system_prompt)} characters)")

        # Load tokenizer and model
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading model (this may take a few minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map=device,
        )
        self.model.eval()

        print("Model loaded successfully!\n")

    def format_prompt(self, user_prompt: str) -> str:
        """Format prompt using Mistral's chat template with system prompt"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Use the tokenizer's chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted

    def generate_response(self, prompt: str) -> str:
        """Generate a single response from the model"""
        try:
            # Format the prompt
            formatted_prompt = self.format_prompt(prompt)

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated tokens (exclude input tokens)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"\n{error_msg}")
            return f"[ERROR: {error_msg}]"

    def load_checkpoint(self, checkpoint_path: Path) -> List[Any]:
        """Load results from checkpoint file"""
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_checkpoint(self, results: List[Any], checkpoint_path: Path):
        """Save current results to checkpoint file"""
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def run_inference(
        self,
        dataset_path: str = "data/safeRLHF_500.json",
        output_path: str = "outputs/mistral_responses.json",
        checkpoint_interval: int = 50,
    ):
        """Run inference on the entire dataset with checkpointing"""

        # Setup paths
        checkpoint_path = Path(output_path).parent / "checkpoint.json"

        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"Dataset loaded: {len(dataset)} prompts\n")

        # Check for existing checkpoint
        results = self.load_checkpoint(checkpoint_path)
        start_idx = len(results)

        if start_idx > 0:
            print(f"Resuming from checkpoint: {start_idx}/{len(dataset)} prompts completed\n")

        # Process prompts
        print("Starting inference...\n")

        for idx in tqdm(range(start_idx, len(dataset)), desc="Processing prompts"):
            prompt_data = dataset[idx]
            prompt = prompt_data["prompt"]

            # Generate response
            response = self.generate_response(prompt)

            # Store result in required format: [[conversation]]
            result = [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            ]
            results.append(result)

            # Save checkpoint periodically
            if (idx + 1) % checkpoint_interval == 0:
                self.save_checkpoint(results, checkpoint_path)
                tqdm.write(f"Checkpoint saved at {idx + 1}/{len(dataset)}")

        # Save final results
        print("\n\nSaving final results...")

        # Save results in required format (nested list structure)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")
        print(f"Format: List of {len(results)} conversations")

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Checkpoint file removed")

        print("\nInference complete!")
        print(f"Processed {len(results)} prompts")


def main():
    """Main entry point"""

    # Configuration
    config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "device": "mps",  # Apple Silicon GPU
        "temperature": 0.7,
        "max_new_tokens": 1500,
        "dataset_path": "data/safeRLHF_500.json",
        "output_path": "outputs/mistral_responses.json",
        "checkpoint_interval": 50,  # Save checkpoint every 50 prompts
    }

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("WARNING: MPS device not available. Falling back to CPU.")
        config["device"] = "cpu"

    # Initialize and run
    runner = MistralInferenceRunner(
        model_name=config["model_name"],
        device=config["device"],
        temperature=config["temperature"],
        max_new_tokens=config["max_new_tokens"],
        system_prompt_path="data/HHH_ICL.txt",
    )

    runner.run_inference(
        dataset_path=config["dataset_path"],
        output_path=config["output_path"],
        checkpoint_interval=config["checkpoint_interval"],
    )


if __name__ == "__main__":
    main()