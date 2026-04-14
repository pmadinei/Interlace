# Evaluation with VLMEvalKit

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for all benchmark evaluations.

## Setup

```bash
pip install vlmeval
```

## Register Models

Copy or merge `vlmevalkit_config.py` into VLMEvalKit's `vlmeval/config.py` to register the INTERLACE pruned models.

## Run Evaluation

```bash
# Evaluate on a single benchmark
python -m vlmeval.run --model Interlace-8B-25pc --data AI2D_TEST

# Evaluate on multiple benchmarks
python -m vlmeval.run --model Interlace-8B-25pc \
    --data AI2D_TEST ChartQA_TEST OCRBench TextVQA_VAL \
    MMBench_DEV_EN_V11 POPE RealWorldQA \
    HRBench4K HRBench8K VStar \
    MIABench ScienceQA_TEST
```

## Benchmarks Used

| Category | Benchmarks |
|----------|-----------|
| Text/Chart | AI2D, ChartQA, OCRBench, TextVQA |
| General VQA | MMBench, POPE, RealWorldQA |
| Perception | HRBench4K, HRBench8K, V-Star |
| Instruction & Science | MIABench, ScienceQA |
