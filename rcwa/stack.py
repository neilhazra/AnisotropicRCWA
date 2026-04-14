from __future__ import annotations

from dataclasses import dataclass, field

from .backend import jnp

from .layer import Layer


@dataclass
class Stack:
    """A stack of x-periodic layers between isotropic substrate and superstrate.

    This class mainly orchestrates the construction of layer and half-space
    RCWA operators. In the current rewrite, the layer-level operator is a full
    matrix in the component-major basis

        [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T

    so any Q matrix returned by this file has shape

        (4 * (2N + 1), 4 * (2N + 1)).
    """

    layers: list[Layer] = field(default_factory=list)
    wavelength_nm: float = 0.0
    kappa_inv_nm: float = 0.0
    eps_substrate: complex = 1.0
    eps_superstrate: complex = 1.0
    _uniform_q_cache: dict[tuple[complex, int, int], jnp.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def period_nm(self) -> float:
        if not self.layers:
            raise ValueError("Stack has no layers")
        return self.layers[0].period_nm

    @property
    def G_normalized(self) -> float:
        return self.wavelength_nm / self.period_nm

    @property
    def kappa_normalized(self) -> float:
        return self.kappa_inv_nm * self.wavelength_nm / (2 * jnp.pi)

    def thickness_normalized(self, layer_index: int) -> float:
        return 2 * jnp.pi * self.layers[layer_index].thickness_nm / self.wavelength_nm

    @staticmethod
    def num_harmonics(N: int) -> int:
        return 2 * N + 1

    @staticmethod
    def harmonic_orders(N: int) -> jnp.ndarray:
        return jnp.arange(-N, N + 1)

    @staticmethod
    def zero_harmonic_index(N: int) -> int:
        return N

    def add_layer(self, layer: Layer) -> None:
        if self.layers and abs(layer.period_nm - self.period_nm) > 1e-10:
            raise ValueError(
                f"Layer period {layer.period_nm} nm does not match stack period {self.period_nm} nm"
            )
        self.layers.append(layer)
        self._uniform_q_cache.clear()

    def layer_Q_matrix_normalized(
        self, layer_index: int, N: int, num_points: int = 512
    ) -> jnp.ndarray:
        """Build one layer's full normalized Q matrix. Output: component-major basis."""
        toeplitz_matrices = self.layers[layer_index].build_toeplitz_fourier_matrices(
            N,
            num_points=num_points,
        )
        return Layer.build_Q_matrix_normalized(
            self.harmonic_orders(N),
            self.harmonic_orders(N),
            self.kappa_normalized,
            self.G_normalized,
            toeplitz_matrices,
            N,
        )

    def layer_Q_matrix_harmonic_major_normalized(
        self, layer_index: int, N: int, num_points: int = 512
    ) -> jnp.ndarray:
        """Build one layer's full normalized Q matrix directly in harmonic-major basis."""
        toeplitz_matrices = self.layers[layer_index].build_toeplitz_fourier_matrices(
            N,
            num_points=num_points,
        )
        return Layer.build_Q_matrix_harmonic_major_normalized(
            self.harmonic_orders(N),
            self.harmonic_orders(N),
            self.kappa_normalized,
            self.G_normalized,
            toeplitz_matrices,
            N,
        )

    def build_all_Q_matrices_normalized(self, N: int, num_points: int = 512) -> list[jnp.ndarray]:
        """Build every layer's Q matrix. Output: component-major basis."""
        return [
            self.layer_Q_matrix_normalized(i, N, num_points=num_points)
            for i in range(len(self.layers))
        ]

    def build_all_Q_matrices_harmonic_major_normalized(
        self, N: int, num_points: int = 512
    ) -> list[jnp.ndarray]:
        """Build every physical-layer Q matrix directly in harmonic-major basis."""
        return [
            self.layer_Q_matrix_harmonic_major_normalized(i, N, num_points=num_points)
            for i in range(len(self.layers))
        ]

    def layer_reduced_to_tangential_field_transform_component_major(
        self, layer_index: int, N: int, num_points: int = 512
    ) -> jnp.ndarray:
        """Input/output: component-major. Maps [-H_y, H_x, E_y, D_x] -> [-H_y, H_x, E_y, E_x]."""
        toeplitz_matrices = self.layers[layer_index].build_toeplitz_fourier_matrices(
            N,
            num_points=num_points,
        )
        return Layer.build_reduced_to_tangential_field_transform_component_major(
            toeplitz_matrices,
            N,
            self.kappa_normalized,
            self.G_normalized,
        )

    def _build_uniform_medium_Q_normalized(
        self,
        eps: complex,
        N: int,
        num_points: int = 512,
    ) -> jnp.ndarray:
        """Build Q for a uniform isotropic medium. Output: component-major basis."""
        cache_key = (complex(eps), N, num_points)
        if cache_key not in self._uniform_q_cache:
            x_domain_nm = self.layers[0].x_domain_nm if self.layers else (0.0, 1.0)
            uniform_layer = Layer.uniform(
                thickness_nm=0.0,
                eps_tensor=jnp.asarray(eps, dtype=jnp.complex128) * jnp.eye(3, dtype=jnp.complex128),
                x_domain_nm=x_domain_nm,
            )
            toeplitz_matrices = uniform_layer.build_toeplitz_fourier_matrices(
                N,
                num_points=num_points,
            )
            self._uniform_q_cache[cache_key] = Layer.build_Q_matrix_normalized(
                self.harmonic_orders(N),
                self.harmonic_orders(N),
                self.kappa_normalized,
                self.G_normalized,
                toeplitz_matrices,
                N,
            )
        return self._uniform_q_cache[cache_key]

    def get_Q_substrate_normalized(self, N: int, num_points: int = 512) -> jnp.ndarray:
        """Return the substrate half-space Q matrix. Output: component-major basis."""
        return self._build_uniform_medium_Q_normalized(
            self.eps_substrate,
            N,
            num_points=num_points,
        )

    def get_Q_superstrate_normalized(self, N: int, num_points: int = 512) -> jnp.ndarray:
        """Return the superstrate half-space Q matrix. Output: component-major basis."""
        return self._build_uniform_medium_Q_normalized(
            self.eps_superstrate,
            N,
            num_points=num_points,
        )

    def _uniform_medium_reduced_to_tangential_field_transform_component_major(
        self,
        eps: complex,
        N: int,
    ) -> jnp.ndarray:
        """Input/output: component-major. Maps [-H_y, H_x, E_y, D_x] -> [-H_y, H_x, E_y, E_x]."""
        num_h = self.num_harmonics(N)
        identity = jnp.eye(num_h, dtype=jnp.complex128)
        zero = jnp.zeros((num_h, num_h), dtype=jnp.complex128)
        inv_eps = identity / jnp.asarray(eps, dtype=jnp.complex128)
        return jnp.block(
            [
                [identity, zero, zero, zero],
                [zero, identity, zero, zero],
                [zero, zero, identity, zero],
                [zero, zero, zero, inv_eps],
            ]
        )

    def substrate_reduced_to_tangential_field_transform_component_major(
        self,
        N: int,
    ) -> jnp.ndarray:
        """Return the substrate reduced-to-tangential field transform in component-major basis."""
        return self._uniform_medium_reduced_to_tangential_field_transform_component_major(
            self.eps_substrate,
            N,
        )

    def superstrate_reduced_to_tangential_field_transform_component_major(
        self,
        N: int,
    ) -> jnp.ndarray:
        """Return the superstrate reduced-to-tangential field transform in component-major basis."""
        return self._uniform_medium_reduced_to_tangential_field_transform_component_major(
            self.eps_superstrate,
            N,
        )
