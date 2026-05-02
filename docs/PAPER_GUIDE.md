# Paper Guide

This repository contains the current IEEE-style paper source under
`paper_ieee/`.

The paper's main claims map to the codebase as follows:

- **Headline Jetson tradeoff**
  - `experiments/exp5_pareto.py`
- **Safety-aware policy comparison**
  - `experiments/exp7_swas.py`
- **Live KITTI VRU recall**
  - `experiments/exp8_accuracy_per_tier.py`
- **Cross-device ONNX deployment**
  - `experiments/exp9_multidevice.py`
  - `experiments/exp9_aggregate.py`

Compile the paper with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

from inside `paper_ieee/`.
