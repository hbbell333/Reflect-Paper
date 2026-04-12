"""
SFT (Supervised Fine-Tuning) training script for the Gemma 2 model family.

Supported models
----------------
  - google/gemma-2-2b-it
  - google/gemma-2-9b-it

What this script does
---------------------
  Trains the model and saves LoRA adapter weights at N evenly-spaced
  checkpoints across the full training run (all epochs combined).
  No KL, no validation — all post-training analysis is in eval_sft_gemma.py.

Checkpoint spacing (Option 2)
------------------------------
  Checkpoints are spaced evenly across total training exposure:

    total_sample_exposures = num_epochs * num_training_samples
    samples_between_ckpts  = total_sample_exposures / num_checkpoints
    steps_between_ckpts    = samples_between_ckpts / (bs * grad_accum)

  Both num_epochs and num_training_samples are derived at runtime — from the
  config file and the actual input data respectively. Nothing is hardcoded.

  Example: 3 epochs, 1600 samples, bs=1, grad_accum=8, 5 checkpoints:
    total_sample_exposures = 3 * 1600 = 4800
    samples_between_ckpts  = 4800 / 5 = 960
    steps_between_ckpts    = 960 / 8  = 120
    checkpoints at steps:  120, 240, 360, 480, 600

Input data format
-----------------
  JSON array (list of objects):
    [{"prompt": "...", "response": "..."}, ...]

  Or JSONL (one object per line):
    {"prompt": "...", "response": "..."}
    {"prompt": "...", "response": "..."}

  An optional "system" field is supported in either format.
  num_training_samples is counted from the actual file at runtime.

Train/val split
---------------
  20% of records are held out for validation before training begins.
  The split is deterministic (controlled by --seed) and stratified by
  shuffle then slice, so the same seed always produces the same split.

  Both splits are saved to disk at the start of each run:
    <output_dir>/train_split.json   <- 80% used for training
    <output_dir>/val_split.json     <- 20% held out (pass to eval script)

  Pass val_split.json as --val_file to eval_sft_gemma.py.

Config file
-----------
  Pass your existing gemma_sft_config.yaml via --config_file.
  All LoRA and training hyperparameters are read from it.
  DPO-specific fields (beta, eval_steps, etc.) are silently ignored.
  The config file is always an explicit CLI argument — no assumed path.

Checkpoint layout
-----------------
  <output_dir>/
    checkpoint_step_<N>/
      adapter_config.json
      adapter_model.safetensors
      tokenizer files
    checkpoint_final/      <- written if last step != a checkpoint boundary

Usage
-----
  python train_sft_gemma.py \\
      --model_size    2b \\
      --train_file    data/train.json \\
      --output_dir    outputs/gemma_2b \\
      --config_file   gemma_sft_config.yaml

  # Both sizes sequentially
  python train_sft_gemma.py \\
      --model_size    all \\
      --train_file    data/train.json \\
      --output_dir    outputs/ \\
      --config_file   gemma_sft_config.yaml

  # Override number of checkpoints
  python train_sft_gemma.py \\
      --model_size       2b \\
      --train_file       data/train.json \\
      --output_dir       outputs/gemma_2b \\
      --config_file      gemma_sft_config.yaml \\
      --num_checkpoints  10
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import fire
import torch
import yaml
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, str] = {
    "2b": "google/gemma-2-2b-it",
    "9b": "google/gemma-2-9b-it",
}

# Gemma 2 projection names match Ministral — existing YAML target_modules
# carries over without changes.
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict:
    """Load a YAML config file.

    Args:
        path: Explicit path passed via --config_file. Never assumed.

    Returns:
        Parsed config dict.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return cfg


def extract_lora_config(yaml_cfg: Dict) -> Dict:
    m = yaml_cfg.get("model", {})
    return {
        "lora_r":         m.get("lora_r",       16),
        "lora_alpha":     m.get("lora_alpha",   32),
        "lora_dropout":   m.get("lora_dropout", 0.1),
        "target_modules": m.get("target_modules", DEFAULT_LORA_TARGET_MODULES),
    }


def extract_training_config(yaml_cfg: Dict) -> Dict:
    """Extract training hyperparameters from YAML.

    DPO-specific fields (beta, eval_steps, save_steps, prediction_loss_only,
    report_to) are not read here and are silently ignored.
    """
    t = yaml_cfg.get("training", {})
    return {
        "num_train_epochs":            t.get("num_train_epochs",            3),
        "per_device_train_batch_size": t.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": t.get("gradient_accumulation_steps", 8),
        "learning_rate":               float(t.get("learning_rate",       5e-5)),
        "warmup_ratio":                t.get("warmup_ratio",              0.1),
        "weight_decay":                t.get("weight_decay",              0.01),
        "max_grad_norm":               t.get("max_grad_norm",              1.0),
        "lr_scheduler_type":           t.get("lr_scheduler_type",       "cosine"),
        "logging_steps":               t.get("logging_steps",              1),
        "gradient_checkpointing":      t.get("gradient_checkpointing",   True),
        "optim":                       t.get("optim",           "adamw_torch"),
        "dataloader_num_workers":      t.get("dataloader_num_workers",    2),
        "dataloader_pin_memory":       t.get("dataloader_pin_memory",    False),
        "remove_unused_columns":       t.get("remove_unused_columns",    False),
    }


# ---------------------------------------------------------------------------
# Data loading — supports both JSON array and JSONL
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[Dict]:
    """Load training data from a JSON array file or a JSONL file.

    Automatically detects the format by inspecting the first non-whitespace
    character of the file:
      - '[' → JSON array  (your actual format: a list of {prompt, response} dicts)
      - '{' → JSONL       (one JSON object per line)

    num_training_samples is len() of the returned list — nothing is hardcoded.

    Args:
        path: Path to the data file.

    Returns:
        List of record dicts, each containing at minimum 'prompt' and 'response'.

    Raises:
        ValueError: If the file is empty or the format cannot be detected.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Data file is empty: {path}")

    if raw[0] == "[":
        # JSON array format: [{"prompt": ..., "response": ...}, ...]
        records = json.loads(raw)
        logger.info(f"Loaded {len(records)} records (JSON array) from {path}")
    else:
        # JSONL format: one object per line
        records = []
        for i, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i}: {e}")
        logger.info(f"Loaded {len(records)} records (JSONL) from {path}")

    if not records:
        raise ValueError(f"No valid records found in {path}")

    return records


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_records(
    records:    List[Dict],
    val_frac:   float = 0.2,
    seed:       int   = 42,
    output_dir: str   = "",
) -> tuple[List[Dict], List[Dict]]:
    """Shuffle and split records into train (80%) and val (20%) sets.

    The split is deterministic: the same seed always produces the same
    train/val partition, so results are reproducible across runs.

    Both splits are written to <output_dir>/train_split.json and
    <output_dir>/val_split.json so the val set can be passed directly
    to eval_sft_gemma.py via --val_file.

    Args:
        records:    Full list of records loaded from the input data file.
        val_frac:   Fraction to hold out for validation (default 0.2 = 20%).
        seed:       Random seed controlling the shuffle.
        output_dir: Directory to save the split files. If empty, files are
                    not saved (useful for testing).

    Returns:
        Tuple of (train_records, val_records).
    """
    import random
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n_val   = max(1, int(len(shuffled) * val_frac))
    val     = shuffled[:n_val]
    train   = shuffled[n_val:]

    logger.info(
        f"Split: {len(train)} train ({100*(1-val_frac):.0f}%) + "
        f"{len(val)} val ({100*val_frac:.0f}%)  [seed={seed}]"
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, split in [("train_split.json", train), ("val_split.json", val)]:
            path = os.path.join(output_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(split, f, indent=2)
            logger.info(f"  Saved {name} → {path}")

    return train, val


# ---------------------------------------------------------------------------
# Checkpoint step calculation — Option 2, fully derived at runtime
# ---------------------------------------------------------------------------

def compute_steps_per_checkpoint(
    num_training_samples: int,
    num_epochs:           int,
    per_device_batch_size: int,
    gradient_accumulation: int,
    num_checkpoints:      int,
) -> int:
    """Compute optimizer steps between checkpoints (Option 2 spacing).

    Checkpoints are evenly spaced across the full training run including
    all epochs, so each checkpoint represents an equal fraction of total
    gradient updates.

    Formula:
        total_sample_exposures = num_epochs * num_training_samples
        samples_per_step       = per_device_batch_size * gradient_accumulation
        total_steps            = ceil(total_sample_exposures / samples_per_step)
        steps_per_checkpoint   = max(1, total_steps // num_checkpoints)

    All inputs are derived at runtime — nothing is hardcoded.

    Args:
        num_training_samples:  Actual count of records in the training file.
        num_epochs:            From config (default 3).
        per_device_batch_size: From config (default 1).
        gradient_accumulation: From config (default 8).
        num_checkpoints:       From CLI --num_checkpoints (default 5).

    Returns:
        Number of optimizer steps between each checkpoint save.
    """
    samples_per_step      = per_device_batch_size * gradient_accumulation
    total_sample_exposures = num_epochs * num_training_samples
    total_steps           = math.ceil(total_sample_exposures / samples_per_step)
    steps_per_ckpt        = max(1, total_steps // num_checkpoints)

    logger.info(
        f"Checkpoint schedule (Option 2):\n"
        f"  Training samples:       {num_training_samples}\n"
        f"  Epochs:                 {num_epochs}\n"
        f"  Total sample exposures: {total_sample_exposures}\n"
        f"  Samples per step:       {samples_per_step}\n"
        f"  Total steps:            {total_steps}\n"
        f"  Num checkpoints:        {num_checkpoints}\n"
        f"  Steps per checkpoint:   {steps_per_ckpt}\n"
        f"  Checkpoint steps:       "
        + str([steps_per_ckpt * (i + 1) for i in range(num_checkpoints)])
    )

    return steps_per_ckpt


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def format_record_as_chat(record: Dict, tokenizer: PreTrainedTokenizerBase) -> str:
    """Format a record using Gemma 2's chat template.

    Gemma 2 does not support a 'system' role in its chat template.
    If a 'system' field is present it is prepended to the user message
    as plain text, which is the approach recommended in the Gemma 2 model card.

    Args:
        record:    Dict with 'prompt', 'response', and optionally 'system'.
        tokenizer: AutoTokenizer loaded from the Gemma 2 model.

    Returns:
        Formatted string ready for tokenisation.
    """
    user_content = record["prompt"]
    if record.get("system"):
        user_content = f"{record['system']}\n\n{user_content}"

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": record["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )


def build_hf_dataset(
    records:   List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    texts = [format_record_as_chat(r, tokenizer) for r in records]
    tokenized = tokenizer(
        texts, truncation=True, max_length=max_length,
        padding=False, return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return Dataset.from_dict(tokenized)


# ---------------------------------------------------------------------------
# Checkpoint callback — saves adapter weights only
# ---------------------------------------------------------------------------

class CheckpointCallback(TrainerCallback):
    """Saves LoRA adapter weights at evenly-spaced optimizer steps.

    steps_per_ckpt is computed by compute_steps_per_checkpoint() before
    training starts, using actual data size and config values. No constants
    are hardcoded here.

    Args:
        output_dir:     Root output directory for this run.
        tokenizer:      Saved alongside the adapter at each checkpoint.
        steps_per_ckpt: Optimizer steps between checkpoint saves.
    """

    def __init__(
        self,
        output_dir:     str,
        tokenizer:      PreTrainedTokenizerBase,
        steps_per_ckpt: int,
    ):
        self.output_dir     = output_dir
        self.tokenizer      = tokenizer
        self.steps_per_ckpt = steps_per_ckpt

    def on_step_end(
        self,
        args:    SFTConfig,
        state:   TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> TrainerControl:
        if state.global_step % self.steps_per_ckpt == 0:
            self._save(state.global_step, model)
        return control

    def on_train_end(
        self,
        args:    SFTConfig,
        state:   TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> TrainerControl:
        """Save a final checkpoint if the last step wasn't a boundary."""
        ckpt_dir = os.path.join(self.output_dir, "checkpoint_final")
        if not os.path.exists(ckpt_dir):
            self._save(state.global_step, model, label="final")
        return control

    def _save(
        self,
        step:  int,
        model: torch.nn.Module,
        label: Optional[str] = None,
    ) -> None:
        label    = label or f"step_{step}"
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint_{label}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Checkpoint saved → {ckpt_dir}")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id:       str,
    lora_r:         int,
    lora_alpha:     int,
    lora_dropout:   float,
    target_modules: List[str],
):
    """Load a Gemma 2 model + tokenizer, wrap with LoRA.

    Key points:
      - AutoTokenizer (not MistralCommonBackend)
      - AutoModelForCausalLM (not Mistral3ForConditionalGeneration)
      - No FineGrainedFP8Config — Gemma 2 weights are BF16 natively
      - attn_implementation="eager" required for Gemma 2's sliding window
        attention to produce correct gradients during training
    """
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Loading model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    base_model.config.use_cache = False

    logger.info(
        f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}, targets={target_modules}"
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    # Otherwise pytorch refuses to create computation graph
    model.enable_input_require_grads()

    return model, tokenizer


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters for one SFT run.

    num_checkpoints is stored here so it flows through to
    compute_steps_per_checkpoint() at runtime alongside the actual
    training sample count.
    """
    model_id:       str
    model_size_key: str
    train_file:     str
    output_dir:     str
    num_checkpoints: int = 5

    # LoRA (from YAML)
    lora_r:          int       = 16
    lora_alpha:      int       = 32
    lora_dropout:    float     = 0.1
    target_modules:  List[str] = field(default_factory=lambda: DEFAULT_LORA_TARGET_MODULES)

    # Training (from YAML)
    num_epochs:             int   = 3
    per_device_train_bs:    int   = 1
    gradient_accumulation:  int   = 8
    learning_rate:          float = 5e-5
    warmup_ratio:           float = 0.1
    weight_decay:           float = 0.01
    max_grad_norm:          float = 1.0
    lr_scheduler_type:      str   = "cosine"
    logging_steps:          int   = 1
    gradient_checkpointing: bool  = True
    optim:                  str   = "adamw_torch"
    dataloader_num_workers: int   = 2
    dataloader_pin_memory:  bool  = False
    remove_unused_columns:  bool  = False

    max_seq_length: int = 3072

    wandb_project: Optional[str] = None
    wandb_name:    Optional[str] = None


# ---------------------------------------------------------------------------
# Single-model training run
# ---------------------------------------------------------------------------

def run_single_model(cfg: TrainConfig) -> None:
    """Execute the full SFT pipeline for one model size.

    Checkpoint spacing is fully derived from actual data count and config —
    no hardcoded sample counts or step counts anywhere in this function.
    """
    wandb_enabled = cfg.wandb_project is not None
    if wandb_enabled:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name or f"sft_{cfg.model_size_key}",
            config=cfg.__dict__,
            reinit=True,
        )

    # Load all records then split 80/20 — val set never touches training
    all_records   = load_records(cfg.train_file)
    train_records, val_records = split_records(
        records=all_records,
        val_frac=0.2,
        seed=42,
        output_dir=cfg.output_dir,
    )
    num_training_samples = len(train_records)
    logger.info(f"Training samples: {num_training_samples}  |  Val samples: {len(val_records)}")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        cfg.model_id, cfg.lora_r, cfg.lora_alpha,
        cfg.lora_dropout, cfg.target_modules,
    )

    # Tokenise training data only — val is kept as raw records for eval script
    train_dataset = build_hf_dataset(train_records, tokenizer, cfg.max_seq_length)

    # Compute checkpoint interval — Option 2, epoch-aware, no hardcoded values
    steps_per_ckpt = compute_steps_per_checkpoint(
        num_training_samples=num_training_samples,
        num_epochs=cfg.num_epochs,
        per_device_batch_size=cfg.per_device_train_bs,
        gradient_accumulation=cfg.gradient_accumulation,
        num_checkpoints=cfg.num_checkpoints,
    )

    callback = CheckpointCallback(
        output_dir=cfg.output_dir,
        tokenizer=tokenizer,
        steps_per_ckpt=steps_per_ckpt,
    )

    # SFTConfig combines TrainingArguments + SFT-specific args (max_length,
    # packing, dataset_text_field) into one object as of TRL 0.15+.
    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_bs,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=True,
        logging_steps=cfg.logging_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        remove_unused_columns=cfg.remove_unused_columns,
        save_strategy="no",
        eval_strategy="no",
        report_to="wandb" if wandb_enabled else "none",
        # SFT-specific args (moved from SFTTrainer into SFTConfig in TRL 0.15+)
        max_length=cfg.max_seq_length,
        packing=False,
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model,
            padding=True, return_tensors="pt", label_pad_token_id=-100,
        ),
        callbacks=[callback],
    )

    logger.info(f"Training {cfg.model_id} …")
    trainer.train()

    if wandb_enabled:
        wandb.finish()

    logger.info(f"✅ {cfg.model_id} done")
    logger.info(f"   Checkpoints: {cfg.output_dir}/checkpoint_*/")
    logger.info(f"   Val split:   {cfg.output_dir}/val_split.json  <- pass to eval_sft_gemma.py --val_file")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    model_size:      str = "2b",
    train_file:      str = "data/train.json",
    output_dir:      str = "outputs/",
    config_file:     Optional[str] = None,
    num_checkpoints: int = 5,

    # CLI fallbacks used only when --config_file is not provided
    num_epochs:            int   = 3,
    per_device_train_bs:   int   = 1,
    gradient_accumulation: int   = 8,
    learning_rate:         float = 5e-5,
    warmup_ratio:          float = 0.1,
    max_seq_length:        int   = 3072,

    wandb_project: Optional[str] = None,
    wandb_name:    Optional[str] = None,
    seed:          int           = 42,
):
    """Train one or both Gemma 2 model sizes with SFT.

    Checkpoints are spaced evenly across the full training run (all epochs),
    using Option 2 spacing. The number of training samples is counted from
    the actual input file — nothing is assumed or hardcoded.

    Args:
        model_size:       '2b', '9b', or 'all'.
        train_file:       Path to training data. Supports JSON array
                          ([{...}, ...]) or JSONL (one object per line).
        output_dir:       Root output directory.
        config_file:      Path to YAML config (e.g. gemma_sft_config.yaml).
                          LoRA and training hyperparameters are read from here.
                          Always passed explicitly — no default path assumed.
        num_checkpoints:  Number of evenly-spaced checkpoints to save.
        wandb_project:    W&B project name. Omit to disable W&B.
        seed:             Random seed for reproducibility.
    """
    set_seed(seed)

    model_size = model_size.lower().strip()
    if model_size not in list(MODEL_REGISTRY) + ["all"]:
        raise ValueError(
            f"--model_size must be one of {list(MODEL_REGISTRY) + ['all']}"
        )

    lora_kw:  Dict = {}
    train_kw: Dict = {}
    if config_file:
        yaml_cfg  = load_yaml_config(config_file)
        lora_kw   = extract_lora_config(yaml_cfg)
        train_kw  = extract_training_config(yaml_cfg)
        logger.info(f"LoRA from YAML:     {lora_kw}")
        logger.info(f"Training from YAML: {train_kw}")

    sizes = list(MODEL_REGISTRY) if model_size == "all" else [model_size]

    for size_key in sizes:
        run_dir = (
            os.path.join(output_dir, f"gemma_{size_key}")
            if model_size == "all" else output_dir
        )
        os.makedirs(run_dir, exist_ok=True)

        cfg = TrainConfig(
            model_id=MODEL_REGISTRY[size_key],
            model_size_key=size_key,
            train_file=train_file,
            output_dir=run_dir,
            num_checkpoints=num_checkpoints,
            wandb_project=wandb_project,
            wandb_name=(
                f"{wandb_name}_{size_key}"
                if wandb_name and model_size == "all" else wandb_name
            ),
            max_seq_length=max_seq_length,
            # LoRA — from YAML if provided, else CLI/dataclass defaults
            lora_r=        lora_kw.get("lora_r",          16),
            lora_alpha=    lora_kw.get("lora_alpha",       32),
            lora_dropout=  lora_kw.get("lora_dropout",    0.1),
            target_modules=lora_kw.get("target_modules",   DEFAULT_LORA_TARGET_MODULES),
            # Training — from YAML if provided, else CLI/dataclass defaults
            num_epochs=            train_kw.get("num_train_epochs",             num_epochs),
            per_device_train_bs=   train_kw.get("per_device_train_batch_size",  per_device_train_bs),
            gradient_accumulation= train_kw.get("gradient_accumulation_steps",  gradient_accumulation),
            learning_rate=         train_kw.get("learning_rate",                learning_rate),
            warmup_ratio=          train_kw.get("warmup_ratio",                 warmup_ratio),
            weight_decay=          train_kw.get("weight_decay",                 0.01),
            max_grad_norm=         train_kw.get("max_grad_norm",                1.0),
            lr_scheduler_type=     train_kw.get("lr_scheduler_type",           "cosine"),
            logging_steps=         train_kw.get("logging_steps",                1),
            gradient_checkpointing=train_kw.get("gradient_checkpointing",      True),
            optim=                 train_kw.get("optim",                "adamw_torch"),
            dataloader_num_workers=train_kw.get("dataloader_num_workers",      2),
            dataloader_pin_memory= train_kw.get("dataloader_pin_memory",       False),
            remove_unused_columns= train_kw.get("remove_unused_columns",       False),
        )

        logger.info(f"\n{'='*60}\n  {size_key.upper()} → {run_dir}\n{'='*60}")
        run_single_model(cfg)

    logger.info("All runs complete.")


if __name__ == "__main__":
    fire.Fire(main)
