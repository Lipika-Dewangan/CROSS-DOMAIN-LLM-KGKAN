# CROSS-DOMAIN-LLM-KGKAN
# LLM-KGKAN : a selective multi-knowledge framework for cross-domain aspect-based sentiment analysis.

## Features
- frozen LLM semantic encoder with LoRA
- dependency-based syntactic encoder
- knowledge graph relational encoder
- KAN-style multi-stream fusion
- asymmetric relational gating
- token-level BIO sentiment tagging
- MMD-based domain alignment

## Project structure
- `config.py`: experiment configuration
- `model.py`: LLM-KGKAN model
- `data.py`: dataset and batch collation
- `train.py`: training script
- `evaluate.py`: evaluation script
- `utils.py`: helper utilities

## Installation
```bash
pip install -r requirements.txt
