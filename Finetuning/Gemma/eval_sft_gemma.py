"""
Post-training evaluation script for Gemma 2 SFT checkpoints.

Loads each saved LoRA adapter checkpoint and runs metrics against
the held-out validation set (never seen during training).

Built-in metric
---------------
  KL divergence (policy || base) — computed on the full val set.
  No labels required; only the model output distributions are compared.

Adding your own metrics
-----------------------
  Fill in run_custom_metrics() near the top of this file.
  Return a flat {metric_name: value} dict.
  Whatever keys you return are merged into the per-checkpoint result.

Output
------
  results/eval_2b.json:
  [
    {
      "checkpoint":     "checkpoint_step_40",
      "checkpoint_dir": "outputs/gemma_2b/checkpoint_step_40",
      "samples_seen":   320,
      "mean_kl":        0.31,
      "median_kl":      0.28,
      "max_kl":         1.12,
      ... keys from run_custom_metrics() ...
    },
    ...
  ]

Usage
-----
  python eval_sft_gemma.py \\
      --checkpoint_dir outputs/gemma_2b \\
      --base_model_id  google/gemma-2-2b-it \\
      --val_file       data/val.jsonl \\
      --output_file    results/eval_2b.json

  python eval_sft_gemma.py \\
      --checkpoint_dir outputs/gemma_2b/checkpoint_step_40 \\
      --base_model_id  google/gemma-2-2b-it \\
      --val_file       data/val.jsonl \\
      --output_file    results/eval_2b_step40.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"checkpoint_step_(\d+)")

# ---------------------------------------------------------------------------
# Pairwise-eval setup (safeRLHF_12 constitution, GPT-4.1 judge via Batch API)
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = Path(__file__).parent
_PAIRWISE_DIR = _SCRIPT_DIR / ".." / "Evaluation" / "pairwise-eval"
_CONSTITUTION = _PAIRWISE_DIR / "Prompts" / "constitutions" / "safeRLHF_12_const.txt"
_DEV_MSG      = _PAIRWISE_DIR / "Prompts" / "pairwise_developer_message.txt"

# Pre-existing base model responses keyed by prompt (no_icl_base_response field)
_BASE_FILE = _SCRIPT_DIR / "data" / "gemma_2B_prompt_base_final.json"

# Cached base-model responses (loaded from disk, reused across all checkpoints)
_BASE_RESPONSES_CACHE: Optional[List[str]] = None

try:
    import dotenv as _dotenv
    _dotenv.load_dotenv()
    sys.path.insert(0, str(_PAIRWISE_DIR))
    from pairwise_eval import (  # type: ignore
        build_cot_request,
        upload_and_run_batch,
        extract_winner_from_reasoning,
    )
    _PAIRWISE_AVAILABLE = True
except ImportError as _err:
    logger.warning(f"Pairwise eval unavailable (import error): {_err}")
    _PAIRWISE_AVAILABLE = False


# ===========================================================================
# Response generation (used by pairwise eval)
# ===========================================================================

@torch.no_grad()
def generate_responses(
    model:          torch.nn.Module,
    tokenizer:      PreTrainedTokenizerBase,
    records:        List[Dict],
    device:         torch.device,
    max_new_tokens: int = 3072,
) -> List[str]:
    """Generate one response per val record using the given model.

    Builds the prompt-only chat string (add_generation_prompt=True) so the
    model continues from the end of the user turn. Only newly generated tokens
    are decoded and returned.

    Args:
        model:          Model to generate from (must already be on device).
        tokenizer:      AutoTokenizer instance.
        records:        Val records; each must have a 'prompt' key.
        device:         Device the model lives on.
        max_new_tokens: Maximum tokens to generate per response.

    Returns:
        List of decoded response strings, one per record.
    """
    model.eval()
    responses = []
    for i, record in enumerate(records):
        if i % 50 == 0:
            logger.info(f"  Generating response {i}/{len(records)} ...")

        user_content = record["prompt"]
        if record.get("system"):
            user_content = f"{record['system']}\n\n{user_content}"

        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    return responses


# ===========================================================================
# ✏️  ADD YOUR METRICS HERE
# ===========================================================================

def run_custom_metrics(
    model:           torch.nn.Module,
    ref_model:       torch.nn.Module,
    tokenizer:       PreTrainedTokenizerBase,
    sequences:       List[str],
    records:         List[Dict],
    device:          torch.device,
    checkpoint_name: str = "",
    output_dir:      str = "",
) -> Dict[str, float]:
    """Pairwise win-rate: SFT checkpoint vs base Gemma 2B on the held-out val set.

    Uses GPT-4.1 as the judge (OpenAI Batch API) with the safeRLHF_12
    constitution.  Base-model responses are generated once and cached to
    <output_dir>/base_responses_cache.json for reuse across checkpoints.

    Full per-pair reasoning is saved to
    <output_dir>/pairwise_<checkpoint_name>.json.

    Returns a flat metrics dict merged into the per-checkpoint result JSON.
    """
    import random

    global _BASE_RESPONSES_CACHE

    if not _PAIRWISE_AVAILABLE:
        logger.warning("Pairwise eval skipped: pairwise_eval dependencies not available.")
        return {}

    if not checkpoint_name or not output_dir:
        logger.warning("Pairwise eval skipped: checkpoint_name or output_dir not provided.")
        return {}

    # ------------------------------------------------------------------
    # 1. Base-model responses (generated once, then cached to disk)
    # ------------------------------------------------------------------
    cache_file = Path(output_dir) / "base_responses_cache.json"

    if _BASE_RESPONSES_CACHE is None:
        if _BASE_FILE.exists():
            logger.info(f"Loading pre-existing base responses from {_BASE_FILE}")
            with open(_BASE_FILE, "r", encoding="utf-8") as f:
                base_data = json.load(f)
            base_by_prompt = {b["prompt"]: b["no_icl_base_response"] for b in base_data}
            _BASE_RESPONSES_CACHE = [base_by_prompt[r["prompt"]] for r in records]
        elif cache_file.exists():
            logger.info(f"Loading cached base responses from {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                _BASE_RESPONSES_CACHE = json.load(f)
        else:
            logger.info("Generating base model responses (no pre-existing file found) ...")
            with model.disable_adapter():
                _BASE_RESPONSES_CACHE = generate_responses(model, tokenizer, records, device)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(_BASE_RESPONSES_CACHE, f, indent=2, ensure_ascii=False)
            logger.info(f"Base responses cached → {cache_file}")

    base_responses = _BASE_RESPONSES_CACHE

    # ------------------------------------------------------------------
    # 2. Checkpoint responses
    # ------------------------------------------------------------------
    logger.info(f"Generating checkpoint responses ({checkpoint_name}) ...")
    checkpoint_responses = generate_responses(model, tokenizer, records, device)

    # ------------------------------------------------------------------
    # 3. Build pairwise cfg from constitution + developer-message files
    # ------------------------------------------------------------------
    with open(_CONSTITUTION, "r", encoding="utf-8") as f:
        constitution = f.read()
    logger.info(f"Constitution loaded: {len(constitution)} chars, first 80: {constitution[:80]!r}")

    with open(_DEV_MSG, "r", encoding="utf-8") as f:
        dev_msg_template = f.read()

    developer_message = dev_msg_template.replace("{CONSTITUTION}", constitution)
    if "{CONSTITUTION}" in developer_message:
        logger.warning("Constitution placeholder was NOT substituted in developer message — check template file")
    else:
        logger.info("Constitution substituted into developer message successfully")

    cfg = {
        "pairwise_constitution": constitution,
        "developer_message": developer_message,
    }

    # ------------------------------------------------------------------
    # 4. Submit OpenAI batch for pairwise CoT judgement
    #    Position A = base, B = checkpoint (randomly flipped per item)
    # ------------------------------------------------------------------
    rng = random.Random(42)
    cot_requests = []
    items_meta   = []

    for i, (record, base_resp, ckpt_resp) in enumerate(
        zip(records, base_responses, checkpoint_responses)
    ):
        prompt = (
            record["prompt"]
            if isinstance(record["prompt"], str)
            else record["prompt"][0]["content"]
        )
        resp_A, resp_B = base_resp, ckpt_resp
        flip = rng.random() < 0.5
        if flip:
            resp_A, resp_B = resp_B, resp_A

        custom_id = f"cot-{i}"
        cot_requests.append(build_cot_request(custom_id, prompt, resp_A, resp_B, cfg))
        items_meta.append({
            "index":               i,
            "custom_id":           custom_id,
            "prompt":              prompt,
            "base_response":       base_resp,
            "checkpoint_response": ckpt_resp,
            "flip_back":           flip,
        })

    logger.info(f"Submitting {len(cot_requests)} pairs to OpenAI Batch API ...")
    cot_results = upload_and_run_batch(cot_requests, f"pairwise-{checkpoint_name}")

    # ------------------------------------------------------------------
    # 5. Extract winners
    #    After flip correction: A = base, B = checkpoint
    # ------------------------------------------------------------------
    evaluations = []
    for meta in items_meta:
        cot_text = cot_results.get(meta["custom_id"])
        if cot_text is None:
            evaluations.append({
                "prompt":              meta["prompt"],
                "base_response":       meta["base_response"],
                "checkpoint_response": meta["checkpoint_response"],
                "eval_reasoning":      None,
                "flipped":             meta["flip_back"],
                "winner":              "Invalid",
            })
            continue

        raw_winner = extract_winner_from_reasoning(cot_text)
        if raw_winner in ("A", "B") and meta["flip_back"]:
            raw_winner = "B" if raw_winner == "A" else "A"

        named = {"A": "base", "B": "checkpoint"}.get(raw_winner, "Invalid")
        evaluations.append({
            "prompt":              meta["prompt"],
            "base_response":       meta["base_response"],
            "checkpoint_response": meta["checkpoint_response"],
            "eval_reasoning":      cot_text,
            "flipped":             meta["flip_back"],
            "winner":              named,
        })

    # ------------------------------------------------------------------
    # 6. Aggregate + save
    # ------------------------------------------------------------------
    total   = len(evaluations)
    ckpt_w  = sum(1 for e in evaluations if e["winner"] == "checkpoint")
    base_w  = sum(1 for e in evaluations if e["winner"] == "base")
    invalid = sum(1 for e in evaluations if e["winner"] == "Invalid")
    valid   = total - invalid

    summary = {
        "checkpoint":         checkpoint_name,
        "total":              total,
        "checkpoint_wins":    ckpt_w,
        "base_wins":          base_w,
        "invalid":            invalid,
        "checkpoint_winrate": ckpt_w / valid if valid > 0 else 0.0,
        "base_winrate":       base_w / valid if valid > 0 else 0.0,
    }

    logger.info(
        f"Pairwise [{checkpoint_name}]: "
        f"checkpoint {summary['checkpoint_winrate']:.2%}  "
        f"base {summary['base_winrate']:.2%}  "
        f"({valid} valid / {invalid} invalid)"
    )

    out_path = Path(output_dir) / f"pairwise_{checkpoint_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([summary] + evaluations, f, indent=2, ensure_ascii=False)
    logger.info(f"Pairwise results → {out_path}")

    return {
        "pairwise_checkpoint_winrate": summary["checkpoint_winrate"],
        "pairwise_base_winrate":       summary["base_winrate"],
        "pairwise_valid":              valid,
        "pairwise_invalid":            invalid,
    }


# ===========================================================================
# Checkpoint discovery
# ===========================================================================

def find_checkpoints(
    checkpoint_dir: str,
    samples_per_step: int,
) -> List[Tuple[str, str, Optional[int]]]:
    """Find all adapter checkpoints under checkpoint_dir.

    Args:
        checkpoint_dir:   Run root or single checkpoint directory.
        samples_per_step: bs * grad_accum from training, used to compute
                          samples_seen labels.

    Returns:
        List of (name, path, samples_seen) sorted by step number.
        samples_seen is None for the 'final' checkpoint.
    """
    p = Path(checkpoint_dir)

    if (p / "adapter_config.json").exists():
        m = _STEP_RE.search(p.name)
        samples = int(m.group(1)) * samples_per_step if m else None
        return [(p.name, str(p), samples)]

    raw = [
        d for d in p.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    ]
    if not raw:
        raise FileNotFoundError(
            f"No adapter checkpoints found under {checkpoint_dir}."
        )

    def sort_key(d: Path) -> int:
        m = _STEP_RE.search(d.name)
        return int(m.group(1)) if m else 10 ** 9

    checkpoints = []
    for d in sorted(raw, key=sort_key):
        m = _STEP_RE.search(d.name)
        samples = int(m.group(1)) * samples_per_step if m else None
        checkpoints.append((d.name, str(d), samples))

    logger.info(f"Found {len(checkpoints)} checkpoints: {[n for n, _, _ in checkpoints]}")
    return checkpoints


# ===========================================================================
# Model loading
# ===========================================================================

def load_base_model(
    model_id: str,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load the frozen Gemma 2 base model on CPU and the tokenizer.

    Differences from the Ministral version:
      - AutoTokenizer replaces MistralCommonBackend.
      - AutoModelForCausalLM replaces Mistral3ForConditionalGeneration.
      - No FineGrainedFP8Config — Gemma 2 weights are BF16 natively.
      - attn_implementation="eager" required for sliding window attention.

    Args:
        model_id: HuggingFace model ID, e.g. 'google/gemma-2-2b-it'.

    Returns:
        (base_model_on_cpu, tokenizer)
    """
    logger.info(f"Loading base model (CPU): {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return base_model, tokenizer


def load_checkpoint_model(
    checkpoint_path: str,
    base_model:      AutoModelForCausalLM,
    device:          torch.device,
) -> PeftModel:
    """Wrap the base model with a saved LoRA adapter checkpoint.

    PeftModel.from_pretrained wraps rather than copies base model weights,
    so only the small adapter tensors are moved to `device`.
    """
    logger.info(f"Loading adapter: {checkpoint_path}")
    policy = PeftModel.from_pretrained(base_model, checkpoint_path)
    policy = policy.to(device)
    policy.eval()
    return policy


# ===========================================================================
# Data helpers
# ===========================================================================

def load_records(path: str) -> List[Dict]:
    """Load records from a JSON array file or a JSONL file.

    Automatically detects format by the first non-whitespace character:
      '[' -> JSON array  (format produced by the training script's val_split.json)
      '{' -> JSONL       (one JSON object per line)

    Args:
        path: Path to the data file.

    Returns:
        List of record dicts with at minimum 'prompt' and 'response' keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Data file is empty: {path}")

    if raw[0] == "[":
        records = json.loads(raw)
    else:
        records = []
        for i, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {i}: {e}")

    if not records:
        raise ValueError(f"No valid records found in {path}")

    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def format_record_as_chat(record: Dict, tokenizer: PreTrainedTokenizerBase) -> str:
    """Format a val record using Gemma 2's chat template.

    Gemma 2 does not support a 'system' role — system content is prepended
    to the first user message as plain text, matching the training script.
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


def build_eval_dataloader(
    sequences:  List[str],
    tokenizer:  PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    """Tokenise val sequences into a DataLoader for KL computation.

    No labels included — KL only needs input_ids and attention_mask.
    """
    tokenized = tokenizer(
        sequences,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    class _DS(TorchDataset):
        def __init__(self, ids, masks):
            self.ids   = ids
            self.masks = masks
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, i):
            return {
                "input_ids":      torch.tensor(self.ids[i]),
                "attention_mask": torch.tensor(self.masks[i]),
            }

    ds = _DS(tokenized["input_ids"], tokenized["attention_mask"])

    def collate(batch):
        return {
            "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


# ===========================================================================
# KL divergence
# ===========================================================================

@torch.no_grad()
def compute_kl_divergence(
    policy_model: torch.nn.Module,
    ref_model:    torch.nn.Module,
    dataloader:   DataLoader,
    device:       torch.device,
) -> Dict[str, float]:
    """Compute token-level KL(policy || ref) over the full val set.

    No ground-truth labels needed — only model output distributions.

    Args:
        policy_model: Fine-tuned checkpoint (on GPU).
        ref_model:    Frozen base model (on CPU). NOTE: PeftModel.from_pretrained
                      mutates base_model in-place, so ref_model is the same
                      mutated object. We disable adapters on policy_model to get
                      clean base logits instead of using ref_model directly.
        dataloader:   Val DataLoader (input_ids + attention_mask, no labels).
        device:       Primary device for policy_model.

    Returns:
        Dict with mean_kl, median_kl, max_kl across all val sequences.
    """
    policy_model.eval()

    per_sample_kls: List[float] = []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        policy_logits = policy_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits

        with policy_model.disable_adapter():
            ref_logits = policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        log_p = F.log_softmax(policy_logits, dim=-1)
        log_q = F.log_softmax(ref_logits,    dim=-1)
        p     = log_p.exp()

        token_kl  = (p * (log_p - log_q)).sum(dim=-1)[:, :-1]
        mask      = attention_mask[:, 1:].float()
        sample_kl = (token_kl * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        per_sample_kls.extend(sample_kl.cpu().float().tolist())

    if not per_sample_kls:
        return {"mean_kl": float("nan"), "median_kl": float("nan"), "max_kl": float("nan")}

    t = torch.tensor(per_sample_kls)
    return {
        "mean_kl":   t.mean().item(),
        "median_kl": t.median().item(),
        "max_kl":    t.max().item(),
    }


# ===========================================================================
# Main evaluation loop
# ===========================================================================

def evaluate_all_checkpoints(
    checkpoint_dir:   str,
    base_model_id:    str,
    val_file:         str,
    output_file:      str,
    max_length:       int = 3072,
    batch_size:       int = 4,
    device_str:       str = "cuda",
    samples_per_step: int = 8,
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Eval device: {device}")

    checkpoints = find_checkpoints(checkpoint_dir, samples_per_step=samples_per_step)
    base_model, tokenizer = load_base_model(base_model_id)

    records   = load_records(val_file)
    sequences = [format_record_as_chat(r, tokenizer) for r in records]
    logger.info(f"Val sequences: {len(sequences)}")

    dataloader = build_eval_dataloader(sequences, tokenizer, max_length, batch_size)
    results    = []

    # Load the first checkpoint to create the PeftModel wrapper once.
    # For subsequent checkpoints we overwrite the adapter weights in-place via
    # load_adapter() to avoid PEFT stacking multiple adapters on the same base.
    first_name, first_path, first_samples = checkpoints[0]
    logger.info(f"\n{'─'*50}")
    logger.info(f"Checkpoint: {first_name}  (samples_seen={first_samples})")
    logger.info(f"{'─'*50}")
    policy_model = load_checkpoint_model(first_path, base_model, device)

    for ckpt_name, ckpt_path, samples_seen in checkpoints:
        if ckpt_path != first_path:
            # Overwrite the single "default" adapter slot with the new checkpoint
            # weights instead of wrapping again (which would stack adapters).
            logger.info(f"\n{'─'*50}")
            logger.info(f"Checkpoint: {ckpt_name}  (samples_seen={samples_seen})")
            logger.info(f"{'─'*50}")
            logger.info(f"Loading adapter: {ckpt_path}")
            policy_model.load_adapter(ckpt_path, adapter_name="default")
            policy_model.set_adapter("default")
            policy_model.eval()

        record: Dict = {
            "checkpoint":     ckpt_name,
            "checkpoint_dir": ckpt_path,
            "samples_seen":   samples_seen,
        }

        logger.info("Computing KL divergence …")
        kl = compute_kl_divergence(policy_model, base_model, dataloader, device)
        record.update(kl)
        logger.info(
            f"  KL — mean={kl['mean_kl']:.4f}  "
            f"median={kl['median_kl']:.4f}  max={kl['max_kl']:.4f}"
        )

        custom = run_custom_metrics(
            model=policy_model,
            ref_model=base_model,
            tokenizer=tokenizer,
            sequences=sequences,
            records=records,
            device=device,
            checkpoint_name=ckpt_name,
            output_dir=checkpoint_dir,
        )
        if custom:
            record.update(custom)
            logger.info(f"  Custom: {custom}")

        results.append(record)

    del policy_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✅ Results written → {output_file}")

    logger.info(
        f"\n{'Checkpoint':<28} {'Samples':>8} {'mean_KL':>10} {'median_KL':>10} {'max_KL':>10}"
    )
    logger.info("─" * 72)
    for r in results:
        logger.info(
            f"{r['checkpoint']:<28} "
            f"{str(r.get('samples_seen', '?')):>8} "
            f"{r.get('mean_kl', float('nan')):>10.4f} "
            f"{r.get('median_kl', float('nan')):>10.4f} "
            f"{r.get('max_kl', float('nan')):>10.4f}"
        )


# ===========================================================================
# Entry point
# ===========================================================================

def main(
    checkpoint_dir:   str,
    base_model_id:    str,
    val_file:         str,
    output_file:      str = "results/eval_results.json",
    max_length:       int = 3072,
    batch_size:       int = 4,
    device:           str = "cuda",
    samples_per_step: int = 8,
):
    """Evaluate Gemma 2 SFT checkpoints on the held-out validation set.

    Args:
        checkpoint_dir:   Run root or single checkpoint directory.
        base_model_id:    e.g. 'google/gemma-2-2b-it'.
        val_file:         Path to held-out val JSONL (never seen in training).
        output_file:      Where to write the JSON results.
        max_length:       Token truncation limit (must match training).
        batch_size:       Eval batch size. Reduce to 2 or 1 if OOM on 9B.
        device:           'cuda' or 'cpu'.
        samples_per_step: per_device_batch_size * gradient_accumulation_steps
                          from training. Annotates samples_seen in output.
    """
    evaluate_all_checkpoints(
        checkpoint_dir=checkpoint_dir,
        base_model_id=base_model_id,
        val_file=val_file,
        output_file=output_file,
        max_length=max_length,
        batch_size=batch_size,
        device_str=device,
        samples_per_step=samples_per_step,
    )


if __name__ == "__main__":
    fire.Fire(main)
