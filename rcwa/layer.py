from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as jnp



FourierCoefficients = dict[str, jnp.ndarray]
FieldQuantities = dict[str, jnp.ndarray]
DielectricTensorFn = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass
class Layer:
    thickness_nm: float
    x_domain_nm: tuple[float, float]
    eps_fn: DielectricTensorFn
    _field_quantities_cache: dict[int, FieldQuantities] = field(default_factory=dict, init=False, repr=False)
    _fourier_coefficients_cache: dict[tuple[int, int], FourierCoefficients] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _toeplitz_cache: dict[tuple[int, int], dict[str, jnp.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def period_nm(self) -> float:
        return self.x_domain_nm[1] - self.x_domain_nm[0]

    def sample_points(self, num_points: int) -> jnp.ndarray:
        """Return one-period sample points used for FFT-based coefficient extraction."""
        return jnp.linspace(
            self.x_domain_nm[0],
            self.x_domain_nm[0] + self.period_nm,
            num_points,
            endpoint=False,
        )

    def eps(self, x_nm: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the dielectric tensor after folding coordinates into one period."""
        x_min_nm, _ = self.x_domain_nm
        x_folded_nm = x_min_nm + (x_nm - x_min_nm) % self.period_nm
        return self.eps_fn(x_folded_nm)

    def sample_eps(self, num_points: int) -> jnp.ndarray:
        """Sample the dielectric tensor on the layer FFT grid."""
        return self.eps(self.sample_points(num_points))

    @staticmethod
    def _field_quantities_from_eps(eps: jnp.ndarray) -> FieldQuantities:
        """Build the reduced dielectric quantities and shorthand compounds."""
        eps_xx = eps[:, 0, 0]
        eps_xy = eps[:, 0, 1]
        eps_xz = eps[:, 0, 2]
        eps_yx = eps[:, 1, 0]
        eps_yy = eps[:, 1, 1]
        eps_yz = eps[:, 1, 2]
        eps_zx = eps[:, 2, 0]
        eps_zy = eps[:, 2, 1]
        eps_zz = eps[:, 2, 2]

        inv_eps_zz = 1.0 / eps_zz

        # These are the z-Schur-complemented in-plane dielectric entries:
        #   \hat{\epsilon}_{ij}(x) = \epsilon_{ij}(x) - \epsilon_{iz}(x)\epsilon_{zz}^{-1}(x)\epsilon_{zj}(x)
        # for i, j in {x, y}.
        field_quantities: FieldQuantities = {
            "hat_eps_xx": eps_xx - eps_xz * inv_eps_zz * eps_zx,
            "hat_eps_xy": eps_xy - eps_xz * inv_eps_zz * eps_zy,
            "hat_eps_yx": eps_yx - eps_yz * inv_eps_zz * eps_zx,
            "hat_eps_yy": eps_yy - eps_yz * inv_eps_zz * eps_zy,
            "inv_eps_zz": inv_eps_zz,
        }

        # This is \hat{\epsilon}_{xx}^{-1}(x), which appears repeatedly in the
        # shorthand compound functions below.
        field_quantities["inv_hat_eps_xx"] = 1.0 / field_quantities["hat_eps_xx"]

        # At z-interfaces the continuous tangential field is E_x, not D_x. In
        # the reduced D-field formulation used here,
        #
        #   D_x(x) = \hat{\epsilon}_{xx}(x) E_x(x)
        #            + \hat{\epsilon}_{xy}(x) E_y(x)
        #            - i \hat{\epsilon}_{xx}(x)
        #              [\hat{\epsilon}_{xx}^{-1}(x)\epsilon_{xz}(x)\epsilon_{zz}^{-1}(x)]
        #              K_x (-H_y)(x)
        #
        # so the interface map uses
        #
        #   E_x(x) = \hat{\epsilon}_{xx}^{-1}(x) D_x(x)
        #            - \hat{\epsilon}_{xx}^{-1}(x)\hat{\epsilon}_{xy}(x) E_y(x)
        #            - [\hat{\epsilon}_{xx}^{-1}(x)\epsilon_{xz}(x)\epsilon_{zz}^{-1}(x)]
        #              K_x (-H_y)(x).
        #
        # We cache the pointwise product directly so its Fourier-convolution
        # matrix is built from the product spectrum rather than from a product
        # of separately truncated Toeplitz matrices.
        field_quantities["inv_hat_eps_xx_hat_eps_xy"] = (
            field_quantities["inv_hat_eps_xx"] * field_quantities["hat_eps_xy"]
        )
        field_quantities["inv_hat_eps_xx_c"] = (
            field_quantities["inv_hat_eps_xx"] * eps_xz * field_quantities["inv_eps_zz"]
        )

        # This is the further-reduced yy entry:
        #   \tilde{\epsilon}_{yy}(x)
        #     = \hat{\epsilon}_{yy}(x)
        #       - \hat{\epsilon}_{yx}(x)\hat{\epsilon}_{xx}^{-1}(x)\hat{\epsilon}_{xy}(x)
        field_quantities["tilde_eps_yy"] = (
            field_quantities["hat_eps_yy"]
            - field_quantities["hat_eps_yx"]
            * field_quantities["inv_hat_eps_xx"]
            * field_quantities["hat_eps_xy"]
        )

        # This is the scalar quantity
        #   \eta(x)
        #     = \epsilon_{zz}^{-1}(x)
        #       + \epsilon_{zz}^{-1}(x)\epsilon_{zx}(x)\hat{\epsilon}_{xx}^{-1}(x)
        #         \epsilon_{xz}(x)\epsilon_{zz}^{-1}(x).
        field_quantities["eta"] = (
            field_quantities["inv_eps_zz"]
            + field_quantities["inv_eps_zz"]
            * eps_zx
            * field_quantities["inv_hat_eps_xx"]
            * eps_xz
            * field_quantities["inv_eps_zz"]
        )

        # These are the shorthand compound functions exactly as written in the derivation:
        #   a(x) = \hat{\epsilon}_{yx}\hat{\epsilon}_{xx}^{-1}\epsilon_{xz}\epsilon_{zz}^{-1}
        #          - \epsilon_{yz}\epsilon_{zz}^{-1}
        field_quantities["a"] = (
            field_quantities["hat_eps_yx"]
            * field_quantities["inv_hat_eps_xx"]
            * eps_xz
            * field_quantities["inv_eps_zz"]
            - eps_yz * field_quantities["inv_eps_zz"]
        )

        #   b(x) = \hat{\epsilon}_{yx}\hat{\epsilon}_{xx}^{-1}
        field_quantities["b"] = (
            field_quantities["hat_eps_yx"] * field_quantities["inv_hat_eps_xx"]
        )

        #   c(x) = \epsilon_{xz}\epsilon_{zz}^{-1}
        field_quantities["c"] = eps_xz * field_quantities["inv_eps_zz"]

        #   d(x) = \epsilon_{zz}^{-1}\epsilon_{zx}\hat{\epsilon}_{xx}^{-1}
        field_quantities["d"] = (
            field_quantities["inv_eps_zz"] * eps_zx * field_quantities["inv_hat_eps_xx"]
        )

        #   e(x) = \epsilon_{zz}^{-1}\left(
        #            \epsilon_{zx}\hat{\epsilon}_{xx}^{-1}\hat{\epsilon}_{xy} - \epsilon_{zy}
        #          \right)
        field_quantities["e"] = field_quantities["inv_eps_zz"] * (
            eps_zx * field_quantities["inv_hat_eps_xx"] * field_quantities["hat_eps_xy"]
            - eps_zy
        )

        return field_quantities

    def field_quantities(self, num_points: int = 512) -> FieldQuantities:
        """Sample and convert the dielectric tensor into the RCWA field quantities."""
        if num_points not in self._field_quantities_cache:
            self._field_quantities_cache[num_points] = self._field_quantities_from_eps(
                self.sample_eps(num_points)
            )
        return self._field_quantities_cache[num_points]

    @staticmethod
    def _fft_centered_coefficients(values: jnp.ndarray, N: int) -> jnp.ndarray:
        """Return Fourier coefficients indexed from -2N to 2N."""
        num_points = values.shape[0]
        fft_vals = jnp.fft.fft(values) / num_points
        pos = fft_vals[: 2 * N + 1]
        neg = fft_vals[-2 * N :]
        return jnp.concatenate([neg, pos])

    def fourier_coefficients(self, N: int, num_points: int = 512) -> FourierCoefficients:
        """Compute Fourier coefficients of the RCWA field quantities used by Q."""
        cache_key = (N, num_points)
        if cache_key not in self._fourier_coefficients_cache:
            quantities = self.field_quantities(num_points=num_points)
            self._fourier_coefficients_cache[cache_key] = {
                key: self._fft_centered_coefficients(values, N)
                for key, values in quantities.items()
            }
        return self._fourier_coefficients_cache[cache_key]

    @staticmethod
    def uniform(
        thickness_nm: float,
        eps_tensor: jnp.ndarray,
        x_domain_nm: tuple[float, float] = (0.0, 1.0),
    ) -> "Layer":
        """Create a spatially uniform layer."""
        eps_tensor = jnp.asarray(eps_tensor, dtype=jnp.complex128)
        return Layer(
            thickness_nm=thickness_nm,
            x_domain_nm=x_domain_nm,
            eps_fn=lambda x_nm, _eps=eps_tensor: jnp.broadcast_to(_eps, (*x_nm.shape, 3, 3)),
        )

    @staticmethod
    def piecewise(
        thickness_nm: float,
        x_domain_nm: tuple[float, float],
        segments: list[tuple[float, float, jnp.ndarray]],
    ) -> "Layer":
        """Create a piecewise-constant layer over one period."""
        boundaries_nm = jnp.array([segment[0] for segment in segments] + [segments[-1][1]])
        tensors = jnp.stack(
            [jnp.asarray(segment[2], dtype=jnp.complex128) for segment in segments]
        )

        def eps_fn(x_nm: jnp.ndarray) -> jnp.ndarray:
            idx = jnp.searchsorted(boundaries_nm, x_nm, side="right") - 1
            idx = jnp.clip(idx, 0, len(segments) - 1)
            return tensors[idx]

        return Layer(thickness_nm=thickness_nm, x_domain_nm=x_domain_nm, eps_fn=eps_fn)

    def build_toeplitz_fourier_matrices(self, N: int, num_points: int = 512) -> dict[str, jnp.ndarray]:
        """Build one Toeplitz Fourier-convolution matrix per field quantity."""
        cache_key = (N, num_points)
        if cache_key not in self._toeplitz_cache:
            fourier_coeffs = self.fourier_coefficients(N, num_points=num_points)
            harmonic_orders = jnp.arange(-N, N + 1)
            difference_indices = (
                harmonic_orders[:, None] - harmonic_orders[None, :] + 2 * N
            ).astype(jnp.int32)

            self._toeplitz_cache[cache_key] = {
                key: coeffs[difference_indices]
                for key, coeffs in fourier_coeffs.items()
            }
        return self._toeplitz_cache[cache_key]

    @staticmethod
    def _assemble_harmonic_major_from_component_blocks(
        blocks: list[list[jnp.ndarray]],
    ) -> jnp.ndarray:
        """Assemble a dense matrix directly in harmonic-major ordering from component blocks."""
        block_tensor = jnp.stack(
            [jnp.stack(row, axis=0) for row in blocks],
            axis=0,
        )
        num_row_blocks, num_col_blocks, num_h_rows, num_h_cols = block_tensor.shape
        return block_tensor.transpose(2, 0, 3, 1).reshape(
            num_h_rows * num_row_blocks,
            num_h_cols * num_col_blocks,
        )

    @staticmethod
    def build_reduced_to_tangential_field_transform_component_major(
        toeplitz_matrices: dict[str, jnp.ndarray],
        N: int,
        kappa_normalized: float,
        G_normalized: float,
    ) -> jnp.ndarray:
        """Input/output: component-major. Maps [-H_y, H_x, E_y, D_x] -> [-H_y, H_x, E_y, E_x]."""
        num_h = 2 * N + 1
        identity = jnp.eye(num_h, dtype=jnp.complex128)
        zero = jnp.zeros((num_h, num_h), dtype=jnp.complex128)
        K_x = Layer.build_K_x_diag_matrix(kappa_normalized, G_normalized, N)
        inv_hat_eps_xx = toeplitz_matrices["inv_hat_eps_xx"]
        inv_hat_eps_xx_hat_eps_xy = toeplitz_matrices["inv_hat_eps_xx_hat_eps_xy"]
        inv_hat_eps_xx_c = toeplitz_matrices["inv_hat_eps_xx_c"]

        blocks = [
            [identity, zero, zero, zero],
            [zero, identity, zero, zero],
            [zero, zero, identity, zero],
            [-inv_hat_eps_xx_c @ K_x, zero, -inv_hat_eps_xx_hat_eps_xy, inv_hat_eps_xx],
        ]
        return jnp.block(blocks)

    @staticmethod
    def build_reduced_to_tangential_field_transform_harmonic_major(
        toeplitz_matrices: dict[str, jnp.ndarray],
        N: int,
        kappa_normalized: float,
        G_normalized: float,
    ) -> jnp.ndarray:
        """Input/output: harmonic-major. Maps [-H_y, H_x, E_y, D_x] -> [-H_y, H_x, E_y, E_x]."""
        num_h = 2 * N + 1
        identity = jnp.eye(num_h, dtype=jnp.complex128)
        zero = jnp.zeros((num_h, num_h), dtype=jnp.complex128)
        K_x = Layer.build_K_x_diag_matrix(kappa_normalized, G_normalized, N)
        inv_hat_eps_xx = toeplitz_matrices["inv_hat_eps_xx"]
        inv_hat_eps_xx_hat_eps_xy = toeplitz_matrices["inv_hat_eps_xx_hat_eps_xy"]
        inv_hat_eps_xx_c = toeplitz_matrices["inv_hat_eps_xx_c"]
        return Layer._assemble_harmonic_major_from_component_blocks(
            [
                [identity, zero, zero, zero],
                [zero, identity, zero, zero],
                [zero, zero, identity, zero],
                [-inv_hat_eps_xx_c @ K_x, zero, -inv_hat_eps_xx_hat_eps_xy, inv_hat_eps_xx],
            ]
        )

    @staticmethod
    def build_K_x_diag_matrix(
        kappa_normalized: float,
        G_normalized: float,
        N: int,
    ) -> jnp.ndarray:
        """Return diag(kappa_normalized + n * G_normalized) for n from -N to N."""
        harmonic_orders = jnp.arange(-N, N + 1)
        kx_values = kappa_normalized + harmonic_orders * G_normalized
        return jnp.diag(kx_values.astype(jnp.complex128))

    def build_Q_matrix_normalized(
        n_vals: jnp.ndarray,
        m_vals: jnp.ndarray,
        kappa_normalized: float,
        G_normalized: float,
        fourier_coeffs_dict: FourierCoefficients,
        N: int,
    ) -> jnp.ndarray:
        """Input/output: component-major basis [-H_y, H_x, E_y, D_x]."""
        _ = (n_vals, m_vals)

        num_h = 2 * N + 1
        identity = jnp.eye(num_h, dtype=jnp.complex128)
        zero = jnp.zeros((num_h, num_h), dtype=jnp.complex128)

        # K_x is diagonal in the harmonic basis with entries
        # kappa_normalized + n * G_normalized for n = -N, ..., N.
        K_x = Layer.build_K_x_diag_matrix(kappa_normalized, G_normalized, N)
        K_x_squared = K_x @ K_x

        # These are the Toeplitz Fourier-convolution matrices for the reduced
        # material quantities that appear in the block operator.
        toeplitz_matrices = fourier_coeffs_dict
        hat_eps_xx = toeplitz_matrices["hat_eps_xx"]
        hat_eps_xy = toeplitz_matrices["hat_eps_xy"]
        tilde_eps_yy = toeplitz_matrices["tilde_eps_yy"]
        eta = toeplitz_matrices["eta"]
        a = toeplitz_matrices["a"]
        b = toeplitz_matrices["b"]
        c = toeplitz_matrices["c"]
        d = toeplitz_matrices["d"]
        e = toeplitz_matrices["e"]

        # The lower-left block is
        #   Q'_41 = -i [hat_eps_xx] + i [hat_eps_xx] K_x [eta] K_x
        # in the normalized convention used by this code.
        Q_prime_41 = -1j * hat_eps_xx + 1j * hat_eps_xx @ K_x @ eta @ K_x

        # Assemble the 4 x 4 block operator acting on the ordered basis
        # [-H_y, H_x, E_y, D_x]^T.
        blocks = [
            [zero, zero, zero, -1j * identity],
            [1j * a @ K_x, zero, 1j * (K_x_squared - tilde_eps_yy), -1j * b],
            [zero, -1j * identity, zero, zero],
            [
                Q_prime_41,
                -1j * hat_eps_xy,
                1j * hat_eps_xx @ K_x @ e,
                -1j * c @ K_x - 1j * hat_eps_xx @ K_x @ d,
            ],
        ]
        return jnp.block(blocks)

    @staticmethod
    def build_Q_matrix_harmonic_major_normalized(
        n_vals: jnp.ndarray,
        m_vals: jnp.ndarray,
        kappa_normalized: float,
        G_normalized: float,
        fourier_coeffs_dict: FourierCoefficients,
        N: int,
    ) -> jnp.ndarray:
        """Input/output: harmonic-major basis [-H_y, H_x, E_y, D_x]."""
        _ = (n_vals, m_vals)

        num_h = 2 * N + 1
        identity = jnp.eye(num_h, dtype=jnp.complex128)
        zero = jnp.zeros((num_h, num_h), dtype=jnp.complex128)
        K_x = Layer.build_K_x_diag_matrix(kappa_normalized, G_normalized, N)
        K_x_squared = K_x @ K_x

        toeplitz_matrices = fourier_coeffs_dict
        hat_eps_xx = toeplitz_matrices["hat_eps_xx"]
        hat_eps_xy = toeplitz_matrices["hat_eps_xy"]
        tilde_eps_yy = toeplitz_matrices["tilde_eps_yy"]
        eta = toeplitz_matrices["eta"]
        a = toeplitz_matrices["a"]
        b = toeplitz_matrices["b"]
        c = toeplitz_matrices["c"]
        d = toeplitz_matrices["d"]
        e = toeplitz_matrices["e"]

        Q_prime_41 = -1j * hat_eps_xx + 1j * hat_eps_xx @ K_x @ eta @ K_x

        return Layer._assemble_harmonic_major_from_component_blocks(
            [
                [zero, zero, zero, -1j * identity],
                [1j * a @ K_x, zero, 1j * (K_x_squared - tilde_eps_yy), -1j * b],
                [zero, -1j * identity, zero, zero],
                [
                    Q_prime_41,
                    -1j * hat_eps_xy,
                    1j * hat_eps_xx @ K_x @ e,
                    -1j * c @ K_x - 1j * hat_eps_xx @ K_x @ d,
                ],
            ]
        )
