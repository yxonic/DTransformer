# DTransformer

## Usage

### Train

DTransformer, ASSIST09:

```bash
python scripts/train.py -m DTransformer -d ASSISTchall_slice -bs 32 -p -cl [-o output/DTransformer_ASSISTchall_slice] [--device cuda]
```

DTransformer, ASSISTchall:

```bash
python scripts/train.py -m DTransformer -d ASSISTchall_slice -bs 32 -p -cl -o [output/DTransformer_ASSISTchall_slice] [--device cuda]
```

### Evaluate
