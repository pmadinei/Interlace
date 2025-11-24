# INTERLACE: Interleaved Layer Pruning and Efficient Adaptation in Large Vision-Language Models

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](https://cvpr.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](link-to-arxiv)

Official implementation of "INTERLACE: Interleaved Layer Pruning and Efficient Adaptation in Large Vision-Language Models" (CVPR 2026).

## Overview

INTERLACE is a novel framework that prunes redundant layers in Vision-Language Models (VLMs) while maintaining performance through sample-efficient finetuning. Our key contributions include:

- **Triplet-based layer analysis**: Identifies local redundancy by evaluating sets of three consecutive layers
- **Strategic freeze-and-finetune approach**: Interleaves frozen anchor layers with fine-tuned layers for stable convergence
- **Exceptional efficiency**: Achieves 88.9% average performance retention after dropping 25% of layers, using only 1% of the FineVision dataset for one epoch

<p align="center">
  <img src="assets/interlace_overview.png" alt="INTERLACE Overview" width="800"/>
</p>

## Key Results

- **94.0%** performance retention at 10% layer pruning
- **86.1%** performance retention at 25% layer pruning
- **28.4%** improvement over alternative pruning methods
- Training on just **1% of FineVision** dataset for **1 epoch**

## Code Release

🚧 **Code Coming Soon!** 🚧

We are currently preparing the code for release. The repository will include:

- Layer importance calculation and triplet selection
- Pruning and model construction scripts
- Fine-tuning pipeline
- Evaluation scripts for all benchmarks

Stay tuned for updates!
