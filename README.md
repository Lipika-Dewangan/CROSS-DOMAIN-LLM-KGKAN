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

`requirements.txt`
```text
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
