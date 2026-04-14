import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-8B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None


@dataclass
class InterlaceArguments:
    pruning_strategy: str = field(
        default="interlace",
        metadata={
            "help": "Layer pruning strategy. Options: interlace, interlace_oa, "
            "interlace_tn, random, consecutive"
        },
    )
    drop_ratio: float = field(
        default=0.25,
        metadata={"help": "Fraction of layers to drop (e.g. 0.10, 0.15, 0.20, 0.25)"},
    )
    hs_pack1_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pack-1 hidden state similarity JSON"},
    )
    hs_pack2_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pack-2 hidden state similarity JSON (fallback)"},
    )
    hs_pack3_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pack-3 hidden state similarity JSON (triplet)"},
    )
