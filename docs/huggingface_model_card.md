---
license: apache-2.0
tags:
  - vision-language
  - pruning
  - layer-pruning
  - qwen3-vl
  - interlace
  - cvpr2025
datasets:
  - HuggingFaceM4/FineVision
base_model: Qwen/Qwen3-VL-8B-Instruct
pipeline_tag: image-text-to-text
---

# INTERLACE: Qwen3-VL-8B with 25% Layer Pruning

This model was pruned using **INTERLACE** ([paper](https://arxiv.org/abs/2511.19676) | [code](https://github.com/pmadinei/Interlace)), a novel framework for efficient layer pruning in Vision-Language Models. It retains **86.1%** of the baseline model's performance while providing **1.18x** Time-To-First-Token speedup.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen3-VL-8B-Instruct |
| Pruning Ratio | 25% (9 of 36 layers removed) |
| Remaining Layers | 27 |
| Hidden Size | 4096 |
| Fine-tuning Data | 1% of FineVision (~240K samples) |
| Fine-tuning Epochs | 1 |
| Method | INTERLACE (triplet-based interleaved pruning) |

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "pmadinei/Interlace-Qwen3-VL-8B-25pc",
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(output[0], skip_special_tokens=True))
```

## Performance

### Relative Performance (% of baseline, CoT enabled)

| Benchmark | Score |
|-----------|-------|
| AI2D | 84.4% |
| ChartQA | 86.5% |
| OCRBench | 86.8% |
| TextVQA | 89.7% |
| MMBench | 83.0% |
| POPE | 99.5% |
| RealWorldQA | 87.2% |
| HRBench4K | 84.0% |
| HRBench8K | 82.1% |
| V-Star | 81.8% |
| MIABench | 85.0% |
| ScienceQA | 83.5% |
| **Overall Average** | **86.1%** |

## All INTERLACE Models

| Model | Drop % | Rel. Perf. |
|-------|--------|------------|
| [Interlace-Qwen3-VL-8B-10pc](https://huggingface.co/pmadinei/Interlace-Qwen3-VL-8B-10pc) | 10% | 94.0% |
| [Interlace-Qwen3-VL-8B-15pc](https://huggingface.co/pmadinei/Interlace-Qwen3-VL-8B-15pc) | 15% | 92.1% |
| [Interlace-Qwen3-VL-8B-20pc](https://huggingface.co/pmadinei/Interlace-Qwen3-VL-8B-20pc) | 20% | 86.9% |
| [Interlace-Qwen3-VL-8B-25pc](https://huggingface.co/pmadinei/Interlace-Qwen3-VL-8B-25pc) | 25% | 86.1% |
| [Interlace-Qwen3-VL-4B-10pc](https://huggingface.co/pmadinei/Interlace-Qwen3-VL-4B-10pc) | 10% | 93.9% |
| [Interlace-Qwen3-VL-4B-15pc](https://huggingface.co/pmadinei/Interlace-Qwen3-VL-4B-15pc) | 15% | 91.9% |
| [Interlace-Qwen3-VL-4B-20pc](https://huggingface.co/pmadinei/Interlace-Qwen3-VL-4B-20pc) | 20% | 88.0% |
| [Interlace-Qwen3-VL-4B-25pc](https://huggingface.co/pmadinei/Interlace-Qwen3-VL-4B-25pc) | 25% | 81.7% |

## Citation

```bibtex
@inproceedings{madinei2025interlace,
  title={INTERLACE: Interleaved Layer Pruning and Efficient Adaptation in Large Vision-Language Models},
  author={Madinei, Parsa and Solgi, Ryan and Wen, Ziqi and Skaza, Jonathan and Eckstein, Miguel and Pedarsani, Ramtin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
