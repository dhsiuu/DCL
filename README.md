# DCL-Torch

This is the PyTorch implementation for our DCL. This project is based on [NeuRec](https://github.com/wubinzzu/NeuRec/tree/v3.x). Thanks to the contributors.

## Environment Requirement

The code runs well under python 3.9. The required packages are as follows:

- pytorch == 1.9.1
- numpy == 1.20.3
- scipy == 1.7.1
- pandas == 1.3.4
- cython == 0.29.24

## Quick Start

First, specify dataset and other hypermeters in configuration file *model_config.ini* and *training_config.ini*.

Then, compline the evaluator of cpp implementation with the following command line:
```bash
python local_compile_setup.py build_ext --inplace
```

Finally, run [main.py](./main.py) in IDE.
