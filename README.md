# DTransformer

Code for _Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer_ (accepted at WWW '23).

Cite this work:

```bibtex
@inproceedings{yin2023tracing,
  author = {Yin, Yu and Dai, Le and Huang, Zhenya and Shen, Shuanghong and Wang, Fei and Liu, Qi and Chen, Enhong and Li, Xin},
  title = {Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer},
  year = {2023},
  isbn = {9781450394161},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3543507.3583255},
  doi = {10.1145/3543507.3583255},
  booktitle = {Proceedings of the ACM Web Conference 2023},
  pages = {855–864},
  numpages = {10},
  keywords = {contrastive learning, knowledge tracing, DTransformer},
  location = {Austin, TX, USA},
  series = {WWW '23}
}
```

## Installation

```bash
# clone the project
git clone git@github.com:yxonic/DTransformer.git
cd DTransformer

# within an existing virtual environment (like conda):
pip install -e .

# or, install with [uv](https://docs.astral.sh/uv/)
uv sync
source .venv/bin/activate
uv pip install -e .
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
