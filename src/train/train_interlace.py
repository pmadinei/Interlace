# Adopted from https://github.com/lm-sys/FastChat.
# Adopted from tatsu-lab@stanford_alpaca.
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict, List, Tuple
import sys
from pathlib import Path
import numpy as np
import random

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from data.data_processor import make_supervised_data_module
from train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    InterlaceArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    for n, p in model.visual.named_parameters():
        p.requires_grad = model_args.tune_mm_vision

    for n, p in model.visual.merger.named_parameters():
        p.requires_grad = model_args.tune_mm_mlp

    for n, p in model.language_model.named_parameters():
        p.requires_grad = model_args.tune_mm_llm
    model.lm_head.requires_grad = model_args.tune_mm_llm


def load_similarity_scores(path: str) -> dict:
    """Load hidden state similarity scores from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    layer_keys = list(data.keys())
    mean_sims = [np.mean(data[k]) for k in layer_keys]
    return {layer_keys[i]: mean_sims[i] for i in range(len(layer_keys))}


def select_layers_interlace(
    total_layers: int,
    n_drops: int,
    hs_pack1_path: str,
    hs_pack3_path: str,
    hs_pack2_path: str = None,
) -> Tuple[List[int], List[int]]:
    """INTERLACE: triplet-based layer selection with frozen anchors.

    Selects triplets of layers ranked by pack-3 similarity. Within each
    triplet, the more redundant of the first two layers (by pack-1 score)
    is dropped and the other is fine-tuned; the third layer is frozen as
    an anchor. Falls back to pack-2 triplets if not enough layers selected.
    """
    sim_single = load_similarity_scores(hs_pack1_path)
    sim_sorted_single = dict(sorted(sim_single.items(), key=lambda x: x[1]))

    sim_triple = load_similarity_scores(hs_pack3_path)
    sorted_triple_keys = [
        int(k)
        for k in sorted(sim_triple, key=lambda x: sim_triple[x], reverse=True)
    ]

    layers_to_drop, layers_to_finetune, assigned = [], [], []

    for layer_idx in sorted_triple_keys:
        if (
            layer_idx not in assigned
            and layer_idx + 1 not in assigned
            and layer_idx + 2 not in assigned
        ):
            opt1, opt2 = layer_idx, layer_idx + 1
            if sim_sorted_single[str(opt1)] < sim_sorted_single[str(opt2)]:
                layers_to_drop.append(opt2)
                layers_to_finetune.append(opt1)
            else:
                layers_to_drop.append(opt1)
                layers_to_finetune.append(opt2)
            assigned.extend([opt1, opt2, layer_idx + 2])
        if len(layers_to_drop) == n_drops:
            break

    if len(layers_to_drop) < n_drops and hs_pack2_path:
        sim_double = load_similarity_scores(hs_pack2_path)
        sorted_double_keys = [
            int(k)
            for k in sorted(sim_double, key=lambda x: sim_double[x], reverse=True)
        ]
        for layer_idx in sorted_double_keys:
            if layer_idx not in assigned and layer_idx + 1 not in assigned:
                opt1, opt2 = layer_idx, layer_idx + 1
                if sim_sorted_single[str(opt1)] < sim_sorted_single[str(opt2)]:
                    layers_to_drop.append(opt2)
                    layers_to_finetune.append(opt1)
                else:
                    layers_to_drop.append(opt1)
                    layers_to_finetune.append(opt2)
                assigned.extend([opt1, opt2])
            if len(layers_to_drop) == n_drops:
                break

    return layers_to_drop, layers_to_finetune


def select_layers_interlace_oa(
    total_layers: int,
    n_drops: int,
    hs_pack3_path: str,
    hs_pack2_path: str = None,
) -> Tuple[List[int], List[int]]:
    """Interlace-OA: ordered assignment within triplets (no individual layer analysis)."""
    sim_triple = load_similarity_scores(hs_pack3_path)
    sorted_triple_keys = [
        int(k)
        for k in sorted(sim_triple, key=lambda x: sim_triple[x], reverse=True)
    ]

    layers_to_drop, layers_to_finetune, assigned = [], [], []

    for layer_idx in sorted_triple_keys:
        if (
            layer_idx not in assigned
            and layer_idx + 1 not in assigned
            and layer_idx + 2 not in assigned
        ):
            layers_to_drop.append(layer_idx)
            layers_to_finetune.append(layer_idx + 1)
            assigned.extend([layer_idx, layer_idx + 1, layer_idx + 2])
        if len(layers_to_drop) == n_drops:
            break

    return layers_to_drop, layers_to_finetune


def select_layers_interlace_tn(
    total_layers: int,
    n_drops: int,
    hs_pack1_path: str,
) -> Tuple[List[int], List[int]]:
    """Interlace-TN: train-next strategy using only individual layer similarities."""
    sim_single = load_similarity_scores(hs_pack1_path)
    sorted_keys = [
        int(k)
        for k in sorted(sim_single, key=lambda x: sim_single[x], reverse=True)
    ]

    layers_to_drop, layers_to_finetune, assigned = [], [], []

    for layer_idx in sorted_keys:
        if layer_idx not in assigned and layer_idx + 1 not in assigned:
            layers_to_drop.append(layer_idx)
            layers_to_finetune.append(layer_idx + 1)
            assigned.extend([layer_idx, layer_idx + 1])
        if len(layers_to_drop) == n_drops:
            break

    return layers_to_drop, layers_to_finetune


def select_layers_random(
    total_layers: int,
    n_drops: int,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Random layer selection for ablation."""
    random.seed(seed)
    all_layers = list(range(total_layers))
    layers_to_drop = random.sample(all_layers, n_drops)
    remained = [l for l in all_layers if l not in layers_to_drop]
    layers_to_finetune = random.sample(remained, n_drops)
    return layers_to_drop, layers_to_finetune


def select_layers_consecutive(
    total_layers: int,
    n_drops: int,
    hs_pack1_path: str,
) -> Tuple[List[int], List[int]]:
    """Consecutive block dropping for ablation."""
    sim_single = load_similarity_scores(hs_pack1_path)
    sorted_keys = [
        int(k)
        for k in sorted(sim_single, key=lambda x: sim_single[x], reverse=True)
    ]

    best_layer = sorted_keys[0]
    layers_to_drop = list(range(best_layer, best_layer + n_drops))
    layers_to_finetune = list(
        range(best_layer + n_drops, best_layer + n_drops + n_drops)
    )
    return layers_to_drop, layers_to_finetune


def select_layers(interlace_args, total_layers: int) -> Tuple[List[int], List[int]]:
    """Dispatch to the appropriate layer selection strategy."""
    n_drops = int(total_layers * interlace_args.drop_ratio)
    strategy = interlace_args.pruning_strategy

    if strategy == "interlace":
        return select_layers_interlace(
            total_layers,
            n_drops,
            interlace_args.hs_pack1_path,
            interlace_args.hs_pack3_path,
            interlace_args.hs_pack2_path,
        )
    elif strategy == "interlace_oa":
        return select_layers_interlace_oa(
            total_layers,
            n_drops,
            interlace_args.hs_pack3_path,
            interlace_args.hs_pack2_path,
        )
    elif strategy == "interlace_tn":
        return select_layers_interlace_tn(
            total_layers, n_drops, interlace_args.hs_pack1_path
        )
    elif strategy == "random":
        return select_layers_random(total_layers, n_drops)
    elif strategy == "consecutive":
        return select_layers_consecutive(
            total_layers, n_drops, interlace_args.hs_pack1_path
        )
    else:
        raise ValueError(
            f"Unknown pruning strategy: {strategy}. "
            "Choose from: interlace, interlace_oa, interlace_tn, random, consecutive"
        )


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, InterlaceArguments)
    )
    model_args, data_args, training_args, interlace_args = (
        parser.parse_args_into_dataclasses()
    )

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_name = model_args.model_name_or_path.lower()
    if "qwen3" in model_name and "moe" in model_name:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_name:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    rank0_print(
        f"Loaded model: {model_args.model_name_or_path} ({model.__class__.__name__})"
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # --- Layer pruning ---
    total_layers = len(model.model.language_model.layers)
    n_drops = int(total_layers * interlace_args.drop_ratio)

    layers_to_drop, layers_to_finetune = select_layers(interlace_args, total_layers)

    layers_to_freeze = [
        i
        for i in range(total_layers)
        if i not in layers_to_finetune + layers_to_drop
    ]

    rank0_print(f"Pruning strategy: {interlace_args.pruning_strategy}")
    rank0_print(f"Drop ratio: {interlace_args.drop_ratio}")
    rank0_print(f"Layers to drop ({len(layers_to_drop)}): {sorted(layers_to_drop)}")
    rank0_print(
        f"Layers to fine-tune ({len(layers_to_finetune)}): {sorted(layers_to_finetune)}"
    )
    rank0_print(
        f"Layers to freeze ({len(layers_to_freeze)}): {sorted(layers_to_freeze)}"
    )

    assert len(layers_to_finetune + layers_to_drop + layers_to_freeze) == total_layers
    assert len(set(layers_to_finetune + layers_to_drop)) == n_drops * 2

    # Remove dropped layers from the model
    kept_layers = [
        layer
        for i, layer in enumerate(model.model.language_model.layers)
        if i not in layers_to_drop
    ]
    model.model.language_model.layers = torch.nn.ModuleList(kept_layers)
    model.config.num_hidden_layers = len(model.model.language_model.layers)
    model.config.text_config.num_hidden_layers = len(
        model.model.language_model.layers
    )

    # Unfreeze only the fine-tuning layers (adjusting for dropped indices)
    original_indices = sorted(list(set(range(total_layers)) - set(layers_to_drop)))
    for i in layers_to_finetune:
        if i in original_indices:
            new_idx = original_indices.index(i)
            for param in model.model.language_model.layers[new_idx].parameters():
                param.requires_grad = True

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
