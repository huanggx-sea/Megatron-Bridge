# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Merge Megatron-Bridge LoRA adapters back into dense model weights (model-agnostic).

The script expects two checkpoints:
1. A **LoRA fine-tuning checkpoint** that contains the adapter weights.
2. A **base/pre-trained checkpoint** that holds the original dense weights.

If the base path is not provided, the script will look for ``run_config.yaml``
inside the LoRA checkpoint and read ``checkpoint.pretrained_checkpoint``.

It works for **any model architecture** supported by ``AutoBridge`` and trained
with Megatron-Bridge's `LoRALinear` wrapper (e.g., Llama, Nemotron, Qwen,
DeepSeek, Phi, etc.).

Usage
-----
CPU-only (single process, no GPU required)::

    python merge_lora.py \
        --lora-checkpoint path/to/finetune_ckpt \
        --hf-model-path   path/to/hf_model \
        --output          path/to/merged_ckpt \
        [--pretrained path/to/base_ckpt] \
        --cpu

GPU with tensor/pipeline/expert parallelism::

    torchrun --nproc_per_node <N> merge_lora.py \
        --lora-checkpoint path/to/finetune_ckpt \
        --hf-model-path   path/to/hf_model \
        --output          path/to/merged_ckpt \
        [--pretrained path/to/base_ckpt] \
        [--tp 1] [--pp 1] [--ep 1]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA, LoRAMerge, VLMLoRA
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.training.checkpointing import (
    _generate_model_state_dict,
    apply_peft_adapter_filter_to_state_dict,
)
from megatron.bridge.training.model_load_save import save_megatron_model
from megatron.bridge.training.utils.checkpoint_utils import read_run_config
from megatron.bridge.utils.common_utils import print_rank_0, resolve_path


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Why this patch exists and why it is safe
# ---------------------------------------------------------------------------
#
# Goal: merge a Megatron-Bridge v0.4.0rc0 SFT LoRA checkpoint into the base
# Qwen3.5-35B-A3B weights to produce a dense reference model for RL (GRPO).
#
# The merge flow is:
#   1. Load base model (dense weights).
#   2. Apply LoRA structure to the model — wraps certain linear layers with
#      adapter slots (A matrix, B matrix), B initialized to zero.
#   3. Load trained A and B values from the SFT checkpoint into those slots.
#   4. W_merged = W_base + scale × B @ A  for each wrapped layer.
#   5. Save the merged model.
#
# The mismatch that causes errors in step 3:
#   Our v0.5.0 Megatron-Bridge LoRA implementation wraps ALL linear layers
#   matching linear_qkv / linear_proj / linear_fc1 / linear_fc2 in the
#   model, including those in the MTP (Multi-Token Prediction) layer
#   (language_model.mtp.layers.*).
#
#   However, the SFT training run was done with v0.4.0rc0, which did NOT
#   apply LoRA to the MTP layer.  As a result, the SFT checkpoint has NO
#   adapter keys for MTP — they simply do not exist in the .metadata file.
#
#   When dist_checkpointing.load() tries to load the adapter weights, two
#   separate validators each raise on these missing MTP keys:
#     (a) MCoreLoadPlanner._validate_global_shapes  — Megatron-Core level
#     (b) create_default_local_load_plan             — PyTorch level
#
# Why skipping the MTP adapter keys is correct:
#   The B matrix is initialized to zero → delta = scale × 0 × A = 0.
#   Skipping the load means those adapters keep their zero initialization,
#   so at merge time: W_merged = W_base + 0 = W_base.
#   MTP was not fine-tuned in SFT, so its weights should remain identical
#   to the base model.  Skipping is semantically correct, not a workaround.
#
# Why we must not skip ALL missing keys (only the expected MTP ones):
#   If a trained layer's adapter key is unexpectedly absent from the checkpoint
#   (e.g. corruption, wrong checkpoint path), we still want an error.
#   The discriminating flag is the MTP module path prefix
#   "language_model.mtp.layers." — any absent key with this prefix is an
#   expected MTP omission; anything else is a real problem.
#
# Why we compute the expected-missing set inside _lenient_validate
# (instead of computing it once up-front before calling dist_checkpointing.load()):
#
# This is a non-obvious design choice. To understand why it matters you need
# to know what dist_checkpointing.load() does internally.
#
# ----- The two key concepts -----
#
# 1. ShardedTensor vs ShardedTensorFactory
#
#    When you build a Megatron sharded state dict (_generate_model_state_dict),
#    the leaves of the nested dict are not all the same type:
#
#    - Most leaves are ShardedTensor objects — direct descriptors saying
#      "this tensor with this key lives in this checkpoint slot with this
#      shape."
#    - Some leaves are ShardedTensorFactory objects — *deferred* descriptors.
#      A factory is essentially a closure that says "when load time comes,
#      call me and I will produce a nested dict of one or more ShardedTensors."
#
#    Factories exist because some logical tensors (e.g. MoE expert weights,
#    grouped-GEMM weights) are stored as multiple physical shards in the
#    checkpoint. A single factory in the in-memory state dict expands into
#    multiple ShardedTensor keys at load time.
#
# 2. apply_factories runs inside dist_checkpointing.load()
#
#    When you call dist_checkpointing.load(sharded_state_dict, ckpt_dir),
#    here is what happens in order:
#
#      dist_checkpointing.load(sharded_state_dict, ckpt_dir)
#       └─ load_preprocess(sharded_state_dict)
#           └─ apply_factories(sharded_state_dict)         ← dict mutates here
#       └─ TorchDistLoadShardedStrategy.load(...)
#           └─ MCoreLoadPlanner._validate_global_shapes(   ← our patch hooks here
#                  metadata,
#                  sharded_tensors=<post-factory flat list>
#              )
#
#    apply_factories walks the dict and replaces every ShardedTensorFactory
#    with the nested ShardedTensors it produces. So between the moment you
#    hand the dict to dist_checkpointing.load and the moment our
#    _lenient_validate is invoked, the dict has been *expanded*: it now
#    contains ShardedTensor keys that did not exist as ShardedTensors a
#    moment earlier.
#
# ----- Why pre-computation fails -----
#
# The earlier (rejected) approach was:
#
#     # BEFORE calling dist_checkpointing.load:
#     expected_missing = {
#         st.key for st in _nested_sharded_tensors(sharded_state_dict)
#         if _MTP_LAYER_KEY_PREFIX in st.key
#            and st.key not in checkpoint_metadata
#     }
#     # then call dist_checkpointing.load with this set stashed somewhere
#
# That walks the *pre*-factory dict and only sees the keys that already exist
# as ShardedTensors. The 12 MTP keys produced by expanding ShardedTensorFactory
# objects (notably the MoE linear_fc1.adapter.linear_out B matrices for experts
# and shared_experts) *don't exist yet* at that point, so they are not in
# expected_missing. When the validator later sees them and they are absent
# from the checkpoint metadata, it raises — even though they are exactly the
# kind of MTP keys we wanted to skip.
#
# By contrast, _lenient_validate receives sharded_tensors as an argument from
# inside the loader — that list is the *post-factory* flat list. Computing
# expected_missing from this argument sees all the factory-expanded keys, so
# it is complete.
#
# ----- The "transformer_layer" vs "mtp_model_layer" detail -----
#
# These are just two different key-string formats for MTP adapter tensors
# that arise in different code paths inside Megatron-Core's MTP module. Both
# contain "language_model.mtp.layers." (our prefix), so the same filter
# catches both. Both formats can appear in the post-factory dict, reinforcing
# the point that you must look at the expanded list to see everything.
#
# ----- The "per-call" / "cross-call contamination" point -----
#
# merge_lora.py calls dist_checkpointing.load() twice:
#   1. Once for the base dense model.
#   2. Once for the LoRA adapters.
#
# Both calls go through our patched _lenient_validate. If we computed
# expected_missing once and cached it on a module-level variable, the value
# from call #1 (which legitimately has *no* expected-missing keys, since the
# base model is fully present in its checkpoint) would leak into call #2, or
# vice-versa. Building the set fresh inside each invocation, from each call's
# own sharded_tensors argument, sidesteps that entirely.
# ---------------------------------------------------------------------------

_MTP_LAYER_KEY_PREFIX = "language_model.mtp.layers."

# Patch MCoreLoadPlanner._validate_global_shapes (validator a) and also clean up
# self.state_dict so create_default_local_load_plan (validator b) does not see the
# expected-missing keys either.  _validate_global_shapes is called from inside
# MCoreLoadPlanner.create_local_plan before the parent DefaultLoadPlanner.create_local_plan
# runs, so removing the keys from self.state_dict here takes effect before validator b.
try:
    from megatron.core.dist_checkpointing.strategies.torch import MCoreLoadPlanner as _MCoreLP

    _orig_validate = _MCoreLP._validate_global_shapes

    def _lenient_validate(self, metadata, sharded_tensors):  # noqa: D103
        from megatron.bridge.utils.common_utils import print_rank_0 as _pr0

        # Convert to list so we can iterate twice (sharded_tensors may be a generator).
        sharded_tensors = list(sharded_tensors)

        # Compute which MTP adapter keys are absent from the checkpoint for this call.
        # We build this per-call (not cached) so that multiple dist_checkpointing.load()
        # calls (e.g. base model load followed by adapter load) don't interfere.
        # Only MTP-layer keys are allowed into this skip set; any other absent key
        # still reaches _orig_validate and raises.
        expected_missing = {
            sh.key
            for sh in sharded_tensors
            if _MTP_LAYER_KEY_PREFIX in sh.key and sh.key not in metadata.state_dict_metadata
        }
        if expected_missing:
            _pr0(
                f"[merge_lora] {len(expected_missing)} MTP adapter key(s) absent "
                "from checkpoint (expected — MTP was not trained with LoRA):\n  "
                + "\n  ".join(sorted(expected_missing))
            )

        present = []
        for sh in sharded_tensors:
            if sh.key in metadata.state_dict_metadata:
                present.append(sh)
            elif sh.key in expected_missing:
                # Expected absence: MTP adapter not trained in SFT.
                # Remove from self.state_dict so PyTorch's create_default_local_load_plan
                # does not raise its own missing-key error on this same key.
                self.state_dict.pop(sh.key, None)
            else:
                # Unexpected missing key — log it, then pass through so original validator raises.
                _pr0(f"[merge_lora] UNEXPECTED absent key: {sh.key!r}")
                _orig_validate(self, metadata, [sh])
        _orig_validate(self, metadata, present)

    _MCoreLP._validate_global_shapes = _lenient_validate
except (ImportError, AttributeError) as _patch_err:
    logger.warning("Could not apply lenient shape-validation patch: %s", _patch_err)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _nested_sharded_tensors(obj: object) -> list[ShardedTensor]:
    """Collect all ShardedTensor leaves from a nested dict structure."""
    if isinstance(obj, ShardedTensor):
        return [obj]
    if isinstance(obj, dict):
        result = []
        for v in obj.values():
            result.extend(_nested_sharded_tensors(v))
        return result
    if isinstance(obj, (list, tuple)):
        result = []
        for v in obj:
            result.extend(_nested_sharded_tensors(v))
        return result
    return []


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge Megatron-Bridge LoRA adapters into base weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lora-checkpoint", required=True, help="LoRA fine-tuning checkpoint directory")
    parser.add_argument("--output", required=True, help="Where to store the merged checkpoint")
    parser.add_argument(
        "--hf-model-path",
        required=True,
        help="HuggingFace model name or local path supplying the config of the architecture.",
    )
    parser.add_argument(
        "--pretrained",
        help="Base (dense) checkpoint. If omitted, resolved from run_config.yaml in the LoRA checkpoint.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logging")

    # Parallelism options
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--cpu", action="store_true", help="Load and merge entirely on CPU (no GPU required)")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_pretrained(lora_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return resolve_path(explicit)
    cfg_path = lora_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("run_config.yaml not found in LoRA checkpoint and --pretrained not supplied")
    cfg = read_run_config(str(cfg_path))
    base = cfg.get("checkpoint", {}).get("pretrained_checkpoint")
    if base is None:
        raise ValueError("pretrained_checkpoint missing in run_config.yaml; pass --pretrained")
    return resolve_path(base)


# -----------------------------------------------------------------------------
# Merge routine
# -----------------------------------------------------------------------------


def merge_lora(
    base_dir: Path,
    lora_dir: Path,
    out_dir: Path,
    hf_model_path: str,
    args: argparse.Namespace,
) -> None:
    """
    Merge LoRA adapter weights back into the base model.

    Args:
        base_dir (Path): Path to the directory containing the base model checkpoint (the dense, pre-trained model).
        lora_dir (Path): Path to the directory containing the LoRA fine-tuned checkpoint.
        out_dir (Path): Path to the directory where the merged model checkpoint should be saved.
        hf_model_path (str): HuggingFace model name or local path to the model architecture/configuration.
        args (argparse.Namespace): Command-line arguments containing parallelism and device settings.

    This routine reconstructs the model architecture from HuggingFace config,
    loads the dense base model weights, then loads the LoRA adapter weights
    (optionally reading LoRA hyperparameters from run_config.yaml), and merges
    the LoRA deltas back into the model weights, resulting in a fully merged checkpoint.
    """
    print_rank_0(f"Loading base model from {base_dir}")
    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)

    model_provider = bridge.to_megatron_provider(load_weights=False)

    print_rank_0(f"Setting Parallelism: TP={args.tp} | PP={args.pp} | EP={args.ep}")
    model_provider.tensor_model_parallel_size = args.tp
    model_provider.pipeline_model_parallel_size = args.pp
    model_provider.expert_model_parallel_size = args.ep
    model_provider.expert_tensor_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    if args.cpu:
        if args.tp != 1 or args.pp != 1 or args.ep != 1:
            logger.warning("TP, PP, and EP must be 1 when using CPU merge. Setting to 1.")
            args.tp = 1
            args.pp = 1
            args.ep = 1
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("gloo")
    model_provider.initialize_model_parallel(seed=0)

    mp_overrides = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": args.pp,
        "expert_model_parallel_size": args.ep,
    }

    # 1) Load base model weights
    model = bridge.load_megatron_model(str(base_dir), mp_overrides=mp_overrides)

    # 2) Patch the model with LoRA adapter *structure* (no weights yet)
    # Load LoRA hyper-parameters from the fine-tuning run_config.yaml so we
    # recreate the exact adapter structure (rank, alpha, etc.) that was used
    # during training. Fallback to defaults when the config is missing.
    peft_cfg: dict = {}
    peft_class = LoRA
    cfg_file = lora_dir / "run_config.yaml"
    if cfg_file.exists():
        try:
            run_cfg_dict = read_run_config(str(cfg_file))
            peft_cfg = run_cfg_dict.get("peft", {}) or {}

            # Determine which PEFT class to use based on _target_ field
            target = peft_cfg.get("_target_", "")
            if "VLMLoRA" in target:
                peft_class = VLMLoRA

            allowed_keys = {
                "target_modules",
                "dim",
                "alpha",
                "dropout",
                "dropout_position",
                "normalize_moe_lora",
                "freeze_language_model",
                "freeze_vision_model",
                "freeze_vision_projection",
            }
            peft_cfg = {k: v for k, v in peft_cfg.items() if k in allowed_keys}
        except Exception as err:
            logger.warning(f"Failed to read LoRA settings from {cfg_file}: {err}. Using defaults.")
    else:
        logger.warning(
            "run_config.yaml not found in LoRA checkpoint; using default LoRA settings for structure patching"
        )

    # Initialize the PEFT object with the loaded hyper-parameters
    print_rank_0(f"Using PEFT class: {peft_class.__name__}")
    lora_peft = peft_class(**peft_cfg)
    model = lora_peft(model, training=False)

    # 3) Load weights from the fine-tuned checkpoint
    print_rank_0(f"Loading LoRA adapter weights from {lora_dir}")
    # Generate full sharded_state_dict describing all model tensors
    sharded_state_dict = _generate_model_state_dict(model, {})
    # Keep only LoRA adapter tensors (and any other trainable parameters) so we don't read unnecessary dense weights.
    sharded_state_dict = apply_peft_adapter_filter_to_state_dict(sharded_state_dict, lora_peft)

    # Remove MTP adapter entries whose keys are absent from the v0.4.0rc0 checkpoint.
    # Two formats coexist:
    #   1. "transformer_layer.*" — factory-expanded ShardedTensors; handled by _lenient_validate
    #   2. "mtp_model_layer.*"   — plain ShardedTensors with allow_shape_mismatch=True; NOT in
    #      flexible_shape_sharded_tensors, so they bypass _lenient_validate but later cause
    #      _restore_dict_types to KeyError when they're absent from the loaded mcore_state_dict.
    # Removing both formats here keeps orig_sharded_state_dict consistent with what actually loads.
    from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory as _STF

    def _remove_mtp_adapter_entries(obj: dict) -> None:
        keys_to_remove = []
        for k, v in obj.items():
            if isinstance(v, (ShardedTensor, _STF)) and _MTP_LAYER_KEY_PREFIX in getattr(v, "key", ""):
                keys_to_remove.append(k)
            elif isinstance(v, dict):
                _remove_mtp_adapter_entries(v)
                if not v:
                    keys_to_remove.append(k)
        for k in keys_to_remove:
            del obj[k]

    _remove_mtp_adapter_entries(sharded_state_dict)
    print_rank_0("[merge_lora] Removed MTP adapter entries from sharded_state_dict (MTP not trained in v0.4.0rc0 SFT)")

    # Load those tensors from the checkpoint directory.
    loaded_sd = dist_checkpointing.load(sharded_state_dict, str(lora_dir))
    # dist_checkpointing.load returns the same nested dict structure; we need the model section
    model_section_key = "model" if "model" in loaded_sd else next(k for k in loaded_sd if k.startswith("model"))
    adapter_sd = loaded_sd[model_section_key]
    # Load adapter weights into the base model (strict=False so missing dense weights are ignored)
    model[0].load_state_dict(adapter_sd, strict=False)

    # 4) Merge adapters
    merge = LoRAMerge()
    merged_model = merge(model[0], training=False)
    for m in merged_model.modules():
        if hasattr(m, "adapter"):
            delattr(m, "adapter")

    # Recursively replace any remaining LoRALinear wrappers with their underlying linear modules
    def _unwrap_lora(module):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.to_wrap)
            else:
                _unwrap_lora(child)

    _unwrap_lora(merged_model)

    out_dir.mkdir(parents=True, exist_ok=True)
    print_rank_0(f"Saving merged checkpoint to {out_dir}")
    save_megatron_model([merged_model], out_dir)

    print_rank_0("Merge complete ✔")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """Main function to merge LoRA adapter weights back into the base model."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    lora_dir = resolve_path(args.lora_checkpoint)
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_dir}")
    base_dir = _resolve_pretrained(lora_dir, args.pretrained)
    if not base_dir.exists():
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {base_dir}")
    try:
        merge_lora(
            base_dir=base_dir,
            lora_dir=lora_dir,
            out_dir=resolve_path(args.output),
            hf_model_path=args.hf_model_path,
            args=args,
        )
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA out of memory during merge. Please rerun this script on CPU by adding the `--cpu` flag.")
        raise SystemExit(1)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
