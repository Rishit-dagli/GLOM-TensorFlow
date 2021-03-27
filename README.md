# GLOM TensorFlow

This Python package attempts to implement GLOM in TensorFlow, which allows advances made by several different groups 
transformers, neural fields, contrastive representation learning, distillation and capsules to be combined. This was 
suggested by Geoffrey Hinton in his paper 
["How to represent part-whole hierarchies in a neural network"](https://arxiv.org/abs/2102.12627).

## Installation

Run the following to install:

```shell script
pip install glom-tf
```

## Developing `glom-tf`

To install `glom-tf`, along with tools you need to develop and test, run the following in your virtualenv:

```shell script
git clone git@github.com:Rishit-dagli/Gradient-Centralization-TensorFlow
# or clone your own fork

pip install -e .[dev]
```

## Citations

```bibtex
@misc{hinton2021represent,
    title   = {How to represent part-whole hierarchies in a neural network}, 
    author  = {Geoffrey Hinton},
    year    = {2021},
    eprint  = {2102.12627},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```