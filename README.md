# DTransformer

Code for _Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer_ (accepted at WWW23).

## Installation

```bash
poetry install
```

## Usage

### Train

Train DTransformer with CL loss:

```bash
python scripts/train.py -m DTransformer -d [assist09,assist17,algebra05,statics] -bs 32 -tbs 32 -p -cl --proj [-o output/DTransformer_assist09] [--device cuda]
```

For more options, run:

```bash
python scripts/train.py -h
```

### Evaluate

Evaluate DTransformer:

```bash
python scripts/test.py -m DTransformer -d [assist09,assist17,algebra05,statics] -bs 32 -p -f [output/best_model.pt] [--device cuda]
```
