"""VLMEvalKit configuration snippet for evaluating INTERLACE models.

Copy or merge this into your VLMEvalKit installation's vlmeval/config.py
to register pruned INTERLACE models for benchmarking.

Usage:
  1. Install VLMEvalKit: pip install vlmeval
  2. Merge the `interlace_models` dict below into the `supported_VLM` dict
     in vlmeval/config.py
  3. Run evaluation:
     python -m vlmeval.run --model Interlace-8B-25pc --data AI2D_TEST
"""

from functools import partial

# Import the Qwen3VL model class from VLMEvalKit
# from vlmeval.vlm.qwen3_vl import Qwen3VLChat

# Register INTERLACE pruned models for evaluation.
# Replace model_path values with your actual checkpoint paths or
# HuggingFace Hub model IDs (e.g., "pmadinei/Interlace-Qwen3-VL-8B-25pc").
interlace_models = {
    # --- Main table models ---
    "Interlace-8B-25pc": dict(
        # cls=Qwen3VLChat,
        model_path="pmadinei/Interlace-Qwen3-VL-8B-25pc",  # or local path
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=16384,
    ),
    "Interlace-8B-20pc": dict(
        model_path="pmadinei/Interlace-Qwen3-VL-8B-20pc",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=16384,
    ),
    "Interlace-8B-15pc": dict(
        model_path="pmadinei/Interlace-Qwen3-VL-8B-15pc",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=16384,
    ),
    "Interlace-8B-10pc": dict(
        model_path="pmadinei/Interlace-Qwen3-VL-8B-10pc",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=16384,
    ),
    "Interlace-4B-25pc": dict(
        model_path="pmadinei/Interlace-Qwen3-VL-4B-25pc",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=16384,
    ),
    "Interlace-4B-20pc": dict(
        model_path="pmadinei/Interlace-Qwen3-VL-4B-20pc",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=16384,
    ),
    "Interlace-4B-15pc": dict(
        model_path="pmadinei/Interlace-Qwen3-VL-4B-15pc",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=16384,
    ),
    "Interlace-4B-10pc": dict(
        model_path="pmadinei/Interlace-Qwen3-VL-4B-10pc",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7,
        max_new_tokens=16384,
    ),
}

# To use: add these to supported_VLM in vlmeval/config.py:
#
#   from functools import partial
#   from vlmeval.vlm.qwen3_vl import Qwen3VLChat
#
#   for name, kwargs in interlace_models.items():
#       supported_VLM[name] = partial(Qwen3VLChat, **kwargs)
