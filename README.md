# RCWA1D

`RCWA1D` is a Python package for one-dimensional rigorous coupled-wave analysis
(RCWA) of layered periodic structures.

This repository contains other RCWA experiments as well, but the package
configuration added here installs only the `RCWA1D` module.

## Install

From the repository root:

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```

## Usage

```python
from rcwa import Layer, Solver, Stack
```

You can also import submodules directly, for example:

```python
from rcwa.visualize import create_stack_xz_profile
```
