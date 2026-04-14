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

Build a stack by creating a `Stack` and adding layers in order:

```python
import jax.numpy as jnp
from rcwa import Layer, Stack

stack = Stack(wavelength_nm=633.0, kappa_inv_nm=0.0, eps_substrate=2.1, eps_superstrate=1.0)
stack.add_layer(Layer.uniform(thickness_nm=120.0, eps_tensor=4.0 * jnp.eye(3), x_domain_nm=(0.0, 400.0)))
stack.add_layer(
    Layer.piecewise(
        thickness_nm=60.0,
        x_domain_nm=(0.0, 400.0),
        segments=[(0.0, 200.0, 1.0 * jnp.eye(3)), (200.0, 400.0, 4.0 * jnp.eye(3))],
    )
)
```

Sample x-z fields with `visualize.py`:

```python
from rcwa.visualize import SUPPORTED_COMPONENTS, create_stack_xz_profile

x_nm, z_nm, field_te_czx, field_tm_czx = create_stack_xz_profile(stack, N=64)
ey_te_zx = field_te_czx[SUPPORTED_COMPONENTS.index("E_y")]
```
