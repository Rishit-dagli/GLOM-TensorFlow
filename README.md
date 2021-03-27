# GLOM TensorFlow [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FGLOM-TensorFlow)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FGLOM-TensorFlow)

[![Flake8 Lint](https://github.com/Rishit-dagli/GLOM-TensorFlow/actions/workflows/flake8-lint.yml/badge.svg)](https://github.com/Rishit-dagli/GLOM-TensorFlow/actions/workflows/flake8-lint.yml)
[![Upload Python Package](https://github.com/Rishit-dagli/GLOM-TensorFlow/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Rishit-dagli/GLOM-TensorFlow/actions/workflows/python-publish.yml)
![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![GitHub stars](https://img.shields.io/github/stars/Rishit-dagli/GLOM-TensorFlow?style=social)](https://github.com/Rishit-dagli/GLOM-TensorFlow/stargazers)
[![GitHub followers](https://img.shields.io/github/followers/Rishit-dagli?label=Follow&style=social)](https://github.com/Rishit-dagli)
[![Twitter Follow](https://img.shields.io/twitter/follow/rishit_dagli?style=social)](https://twitter.com/intent/follow?screen_name=rishit_dagli)

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
git clone https://github.com/Rishit-dagli/GLOM-TensorFlow.git
# or clone your own fork

cd GLOM-TensorFlow
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