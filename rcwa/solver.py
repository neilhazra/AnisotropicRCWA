from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .backend import jnp, linalg

from .stack import Stack


ScatteringMatrix = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


class Solver:
    """Modal RCWA solver using adjacent-port scattering matrices."""
    @staticmethod
    def _log(verbose: bool, message: str) -> None:
        """Print a solver progress message when verbose output is enabled."""
        if verbose:
            print(f"[Solver] {message}")

    @staticmethod
    def component_to_harmonic_major(matrix: jnp.ndarray) -> jnp.ndarray:
        """Reorder a full 4*num_h square matrix from component-major to harmonic-major."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 4 != 0:
            raise ValueError(f"Expected a square matrix with size divisible by 4, got {matrix.shape}.")

        num_h = matrix.shape[0] // 4
        return matrix.reshape(4, num_h, 4, num_h).transpose(1, 0, 3, 2).reshape(matrix.shape)

    @staticmethod
    def harmonic_to_component_major_rows(matrix: jnp.ndarray) -> jnp.ndarray:
        """Reorder matrix rows from harmonic-major into component-major field ordering."""
        if matrix.ndim != 2 or matrix.shape[0] % 4 != 0:
            raise ValueError(
                f"Expected a rank-2 matrix with row count divisible by 4, got {matrix.shape}."
            )

        num_h = matrix.shape[0] // 4
        row_idx = jnp.array(
            [4 * h + p for p in range(4) for h in range(num_h)],
            dtype=jnp.int32,
        )
        return matrix[row_idx, :]

    @staticmethod
    def zero_order_mode_index(N: int, incident_pol: str) -> int:
        """Return the forward zero-order modal index for TE or TM incidence."""
        pol = incident_pol.upper()
        zero = Stack.zero_harmonic_index(N)
        if pol == "TE":
            return zero
        if pol == "TM":
            return Stack.num_harmonics(N) + zero
        raise ValueError(f"Unknown incident_pol={incident_pol!r}")

    @staticmethod
    def _harmonic_diag_blocks_block_diagonal(
        Q: jnp.ndarray,
    ) -> jnp.ndarray | None:
        """Extract per-harmonic 4x4 diagonal blocks from a harmonic-major Q matrix."""
        num_h = Q.shape[0] // 4
        q_blocks = Q.reshape(num_h, 4, num_h, 4)
        # harmonic_blocks = q_blocks.transpose(0, 2, 1, 3)
        # block_idx = jnp.arange(num_h)
        # diag_blocks = harmonic_blocks[block_idx, block_idx]
        diag_blocks = jnp.diagonal(q_blocks, axis1=0, axis2=2).transpose(2, 0, 1)
        return diag_blocks

    @staticmethod
    def _block_diagonal_matrix(blocks: jnp.ndarray) -> jnp.ndarray:
        """Assemble a dense block-diagonal matrix from blocks[block, row, col]."""
        return linalg.block_diag(*blocks)

    @staticmethod
    def _normalize_columns(vectors: jnp.ndarray, tol: float = 1e-14) -> jnp.ndarray:
        norms = jnp.linalg.norm(vectors, axis=-2, keepdims=True)
        safe_norms = jnp.where(norms > tol, norms, 1.0)
        return vectors / safe_norms

    @staticmethod
    def _resolve_isotropic_pair(
        v1: jnp.ndarray,
        v2: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Resolve a degenerate isotropic mode pair into a stable TE/TM basis.

        The inputs may be a single pair with shape ``(4,)`` or a batched set of
        pairs with shape ``(..., 4)``.
        """
        w_te = v2[..., 3, None] * v1 - v1[..., 3, None] * v2
        w_tm = v2[..., 2, None] * v1 - v1[..., 2, None] * v2
        pair = jnp.stack([w_te, w_tm], axis=-1)
        pair = Solver._normalize_columns(pair)
        return pair[..., 0], pair[..., 1]

    @staticmethod
    def transfer_to_scattering(T: jnp.ndarray) -> ScatteringMatrix:
        """Convert a 2x2-block transfer matrix into an S-matrix."""
        half = T.shape[0] // 2
        T11 = T[:half, :half]
        T12 = T[:half, half:]
        T21 = T[half:, :half]
        T22 = T[half:, half:]

        rhs = jnp.concatenate([T21, jnp.eye(half, dtype=T22.dtype)], axis=1)
        solution = jnp.linalg.solve(T22, rhs)
        T22_inv_T21 = solution[:, :half]
        T22_inv = solution[:, half:]

        S11 = -T22_inv_T21
        S12 = T22_inv
        S21 = T11 - T12 @ T22_inv_T21
        S22 = T12 @ T22_inv
        return S11, S12, S21, S22

    @staticmethod
    def basis_change_scattering_matrix(
        left_fields: jnp.ndarray,
        right_fields: jnp.ndarray,
    ) -> ScatteringMatrix:
        """Return the zero-thickness S-matrix for a change of adjacent modal basis."""
        return Solver.transfer_to_scattering(
            Solver.basis_change_transfer_matrix(left_fields, right_fields)
        )

    @staticmethod
    def basis_change_transfer_matrix(
        left_fields: jnp.ndarray,
        right_fields: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the coefficient map from the left modal basis to the right modal basis."""
        return jnp.linalg.solve(right_fields, left_fields)

    @staticmethod
    def modal_propagation_scattering_matrix(
        eigenvalues: jnp.ndarray,
        thickness: float,
    ) -> ScatteringMatrix:
        """Return the diagonal propagation S-matrix inside one layer's modal basis.
        Scattering-port convention:
            [a_L^-]   [S11  S12] [a_L^+]
            [a_R^+] = [S21  S22] [a_R^-]
        """
        n = eigenvalues.shape[0]
        if n % 2 != 0:
            raise ValueError(f"Expected an even number of eigenvalues, got shape[0]={n}.")

        half = n // 2
        X_forward = jnp.diag(jnp.exp(eigenvalues[:half] * thickness))
        X_backward = jnp.diag(jnp.exp(-eigenvalues[half:] * thickness))
        Z = jnp.zeros_like(X_forward)
        return Z, X_backward, X_forward, Z

    @staticmethod
    def redheffer_star_product(Sa: ScatteringMatrix, Sb: ScatteringMatrix) -> ScatteringMatrix:
        """Redheffer star product for two compatible adjacent-port S-matrices."""
        A11, A12, A21, A22 = Sa
        B11, B12, B21, B22 = Sb
        half = A11.shape[0]
        I = jnp.eye(half, dtype=A11.dtype)

        system = I - A22 @ B11
        rhs = jnp.concatenate([A21, A22 @ B12], axis=1)
        solution = jnp.linalg.solve(system, rhs)
        inv_a_A21 = solution[:, :half]
        inv_a_A22_B12 = solution[:, half:]

        B11_inv_a_A21 = B11 @ inv_a_A21
        B11_inv_a_A22_B12 = B11 @ inv_a_A22_B12
        B21_inv_a_A21 = B21 @ inv_a_A21
        B21_inv_a_A22_B12 = B21 @ inv_a_A22_B12

        C11 = A11 + A12 @ B11_inv_a_A21
        C12 = A12 @ (B12 + B11_inv_a_A22_B12)
        C21 = B21_inv_a_A21
        C22 = B22 + B21_inv_a_A22_B12
        return C11, C12, C21, C22

    @staticmethod
    def chain_scattering_matrices(S_list: list[ScatteringMatrix]) -> ScatteringMatrix:
        """Chain a list of compatible S-matrices with the Redheffer star product."""
        if not S_list:
            raise ValueError("Expected at least one scattering matrix to chain.")

        result = S_list[0]
        for S in S_list[1:]:
            result = Solver.redheffer_star_product(result, S)
        return result

    @staticmethod
    def identity_scattering_matrix(num_modes: int) -> ScatteringMatrix:
        """Return the transparent two-port S-matrix on one modal basis."""
        identity = jnp.eye(num_modes, dtype=jnp.complex128)
        zero = jnp.zeros_like(identity)
        return zero, identity, identity, zero

    @staticmethod
    def _isotropic_halfspace_mode_to_field(Q_halfspace: jnp.ndarray, N: int, verbose: bool = False) -> jnp.ndarray:
        """Input Q: component-major. Output rows: harmonic-major [-H_y, H_x, E_y, D_x].
        Output columns: [FTE(-N..N), FTM(-N..N), BTE(-N..N), BTM(-N..N)]."""
        num_h = Stack.num_harmonics(N)
        Solver._log(verbose, f"  Extracting {num_h} diagonal 4x4 blocks from harmonic-major Q")
        Q_halfspace_harmonic = Solver.component_to_harmonic_major(Q_halfspace)
        Q_blocks_iso = Solver._harmonic_diag_blocks_block_diagonal(Q_halfspace_harmonic)
        Solver._log(verbose, f"  Block diagonalization: batched eig on {num_h} independent 4x4 blocks (NOT full {4*num_h}x{4*num_h})")
        eigvals, eigvecs = jnp.linalg.eig(Q_blocks_iso)

        # Classify raw isotropic half-space modes by solver-wide propagation
        # direction. "Forward" means propagating or decaying toward +z.
        forward = (jnp.real(eigvals) < -1e-9) | (
            (jnp.abs(jnp.real(eigvals)) <= 1e-9) & (jnp.imag(eigvals) > 0)
        )

        # Bring the forward pair to columns 0:2 and the backward pair to
        # columns 2:4 within each harmonic block.
        idx = jnp.argsort(-forward.astype(jnp.int32), axis=-1)
        eigvals = jnp.take_along_axis(eigvals, idx, axis=-1)
        eigvecs = jnp.take_along_axis(eigvecs, idx[..., None, :], axis=-1)

        # Resolve the isotropic TE/TM degeneracy separately in the forward and
        # backward pairs. The resulting per-harmonic column order is
        # [FTE, FTM, BTE, BTM].
        FTE, FTM = Solver._resolve_isotropic_pair(eigvecs[:, :, 0], eigvecs[:, :, 1])
        BTE, BTM = Solver._resolve_isotropic_pair(eigvecs[:, :, 2], eigvecs[:, :, 3])
        halfspace_blocks = jnp.stack([FTE, FTM, BTE, BTM], axis=-1)

        # Build a block-diagonal fields matrix with rows already in harmonic-major
        # order. block_diag gives columns in block order
        # [FTE(h), FTM(h), BTE(h), BTM(h)] for each harmonic h, so reorder the
        # columns into the solver-wide modal layout
        # [FTE(-N..N), FTM(-N..N), BTE(-N..N), BTM(-N..N)].
        Solver._log(verbose, f"  Assembling block-diagonal mode-to-field matrix ({4*num_h}x{4*num_h}) from {num_h} blocks")
        mode_to_field_halfspace = Solver._block_diagonal_matrix(halfspace_blocks)
        modal_reorder = jnp.array(
            [4 * h + 0 for h in range(num_h)]
            + [4 * h + 1 for h in range(num_h)]
            + [4 * h + 2 for h in range(num_h)]
            + [4 * h + 3 for h in range(num_h)],
            dtype=jnp.int32,
        )
        return mode_to_field_halfspace[:, modal_reorder]

    @staticmethod
    def get_substrate_mode_to_field(stack: Stack, N: int, num_points: int = 512, verbose: bool = False):
        """Return the isotropic substrate modes-to-fields matrix. Rows: harmonic-major."""
        return Solver._isotropic_halfspace_mode_to_field(
            stack.get_Q_substrate_normalized(N, num_points),
            N,
            verbose=verbose,
        )

    @staticmethod
    def get_superstrate_mode_to_field(stack: Stack, N: int, num_points: int = 512, verbose: bool = False):
        """Return the isotropic superstrate modes-to-fields matrix. Rows: harmonic-major."""
        return Solver._isotropic_halfspace_mode_to_field(
            stack.get_Q_superstrate_normalized(N, num_points),
            N,
            verbose=verbose,
        )

    @staticmethod
    def _sort_modes_by_propagation(
        eigvals: jnp.ndarray,
        eigvecs: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sort modes so forward/decaying +z solutions come first."""
        evanescent_comp = jnp.real(eigvals)
        propagating_comp = -jnp.imag(eigvals)
        # Decaying into +z is definitely in the "forward region" since it will
        # be negative. In the -iwt convention propagating in +z is positive, so
        # flip the sign so positive imag goes first.
        sorter = jnp.where(jnp.abs(evanescent_comp) > 1e-9, evanescent_comp, propagating_comp)
        idx = jnp.argsort(sorter, axis=-1)
        return (
            jnp.take_along_axis(eigvals, idx, axis=-1),
            jnp.take_along_axis(eigvecs, idx[..., None, :], axis=-1),
        )

    @staticmethod
    def _diagonalize_sort_dense_layer_system(q_layer: jnp.ndarray, verbose: bool = False):
        """Diagonalize a general layer Q matrix with a dense eigensolve."""
        n = q_layer.shape[0]
        Solver._log(verbose, f"  Full eigendecomposition: eig on {n}x{n} matrix (NOT block diagonal)")
        eig_val, eig_vec = jnp.linalg.eig(q_layer)
        return Solver._sort_modes_by_propagation(eig_val, eig_vec)

    @staticmethod
    def _diagonalize_sort_homogeneous_layer_system(q_layer: jnp.ndarray, verbose: bool = False):
        """Diagonalize a homogeneous layer via its harmonic 4x4 block structure."""
        n = q_layer.shape[0]
        if q_layer.ndim != 2 or q_layer.shape[0] != q_layer.shape[1] or n % 4 != 0:
            raise ValueError(f"Expected a square layer Q matrix with size divisible by 4, got {q_layer.shape}.")

        num_h = n // 4
        Solver._log(
            verbose,
            f"  Homogeneous layer block diagonalization: batched eig on {num_h} independent 4x4 blocks (NOT full {n}x{n})",
        )
        q_layer_harmonic = Solver.component_to_harmonic_major(q_layer)
        q_blocks = Solver._harmonic_diag_blocks_block_diagonal(q_layer_harmonic)
        eigvals_blocks, eigvecs_blocks = jnp.linalg.eig(q_blocks)
        eigvals_blocks, eigvecs_blocks = Solver._sort_modes_by_propagation(
            eigvals_blocks,
            eigvecs_blocks,
        )

        mode_to_field_harmonic = Solver._block_diagonal_matrix(eigvecs_blocks)
        modal_reorder = jnp.array(
            [4 * h + m for h in range(num_h) for m in range(2)]
            + [4 * h + m for h in range(num_h) for m in range(2, 4)],
            dtype=jnp.int32,
        )
        eigvals = jnp.concatenate(
            [
                eigvals_blocks[:, :2].reshape(-1),
                eigvals_blocks[:, 2:].reshape(-1),
            ],
            axis=0,
        )
        eigvecs = Solver.harmonic_to_component_major_rows(
            mode_to_field_harmonic[:, modal_reorder]
        )
        return eigvals, eigvecs

    @staticmethod
    def diagonalize_sort_layer_system(
        q_layer,
        verbose: bool = False,
        is_homogeneous: bool = False,
    ):
        """Diagonalize a layer Q matrix and order forward modes before backward modes."""
        if is_homogeneous:
            return Solver._diagonalize_sort_homogeneous_layer_system(q_layer, verbose=verbose)
        return Solver._diagonalize_sort_dense_layer_system(q_layer, verbose=verbose)

    @staticmethod
    def total_scattering_matrix(
        stack: Stack,
        N: int,
        num_points: int = 512,
        verbose: bool = False,
    ) -> ScatteringMatrix:
        """Return the stack S-matrix in substrate/superstrate modal bases.
        Convention is """
        num_h = Stack.num_harmonics(N)
        mat_size = 4 * num_h
        n_layers = len(stack.layers)
        Solver._log(verbose, f"Building Q matrices for {n_layers} layer(s) (matrix size {mat_size}x{mat_size})")
        layer_qs = stack.build_all_Q_matrices_normalized(N, num_points, verbose=verbose)
        Solver._log(verbose, f"Diagonalizing substrate halfspace (block diagonalization, {num_h} independent 4x4 blocks)")
        substrate_mode_to_field = Solver.harmonic_to_component_major_rows(
            Solver.get_substrate_mode_to_field(stack, N, num_points, verbose=verbose)
        )
        substrate_tang = stack.substrate_reduced_to_tangential_field_transform_component_major(N)
        in_field = substrate_mode_to_field
        in_field_tang = substrate_tang
        S_total = None
        for i, (layer, q_layer) in enumerate(zip(stack.layers, layer_qs)):
            Solver._log(verbose, f"--- Layer {i + 1}/{n_layers} ---")
            layer_tang = stack.layer_reduced_to_tangential_field_transform_component_major(i, N, num_points)
            eigvals, layer_field = Solver.diagonalize_sort_layer_system(
                q_layer,
                verbose=verbose,
                is_homogeneous=layer.is_homogeneous,
            )
            Solver._log(verbose, f"  Computing interface transfer matrix (linear solve, {mat_size}x{mat_size})")
            #TMInterface = jnp.linalg.inv(layer_field) @ jnp.linalg.inv(layer_tang) @ in_field_tang @ in_field
            TMInterface = jnp.linalg.solve(
                                            layer_tang @ layer_field,
                                            in_field_tang @ in_field,
                                        )
            Solver._log(verbose, f"  Converting transfer matrix to scattering matrix")
            S_Mat_interface = Solver.transfer_to_scattering(TMInterface)
            Solver._log(verbose, f"  Computing modal propagation S-matrix")
            Modal_prop = Solver.modal_propagation_scattering_matrix(eigvals, stack.thickness_normalized(i))
            Solver._log(verbose, f"  Redheffer star product: interface x propagation")
            S_layer = Solver.redheffer_star_product(S_Mat_interface, Modal_prop)
            Solver._log(verbose, f"  Redheffer star product: accumulating total S-matrix")
            S_total = S_layer if S_total is None else Solver.redheffer_star_product(S_total, S_layer)
            in_field = layer_field
            in_field_tang = layer_tang

        Solver._log(verbose, f"Diagonalizing superstrate halfspace (block diagonalization, {num_h} independent 4x4 blocks)")
        out_field = Solver.harmonic_to_component_major_rows(
            Solver.get_superstrate_mode_to_field(stack, N, num_points, verbose=verbose)
        )
        out_tang = stack.superstrate_reduced_to_tangential_field_transform_component_major(N)
        Solver._log(verbose, f"Computing final superstrate interface (matrix inversion + multiply, {mat_size}x{mat_size})")
        TMInterface = jnp.linalg.inv(out_field) @ jnp.linalg.inv(out_tang) @ in_field_tang @ in_field
        S_Mat_interface = Solver.transfer_to_scattering(TMInterface)
        Solver._log(verbose, "Redheffer star product: accumulating final interface")
        S_total = S_Mat_interface if S_total is None else Solver.redheffer_star_product(S_total, S_Mat_interface)
        Solver._log(verbose, "Total scattering matrix complete")

        return (
            S_total,
            substrate_mode_to_field, substrate_tang,
            out_field, out_tang,
        )


    @staticmethod
    def reflection_transmission(
        stack: Stack,
        N: int,
        num_points: int = 512,
        verbose: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (r, t) E-field ratio matrices. Rows: [Ey(-N..N), Ex(-N..N)]. Cols: [TE(-N..N), TM(-N..N)]."""
        Solver._log(verbose, f"Computing reflection/transmission for N={N}, {len(stack.layers)} layer(s)")
        (S11, _, S21, _), in_f, i_f_t, o_f, o_f_t = Solver.total_scattering_matrix(
            stack,
            N,
            num_points=num_points,
            verbose=verbose,
        )

        half = S11.shape[0]
        num_h = Stack.num_harmonics(N)

        Solver._log(verbose, f"Extracting E-field ratios (matrix multiplications, {4*num_h}x{half})")
        # Full tangential-field matrices: (4*num_h, 4*num_h)
        # Columns split as [forward_modes | backward_modes], each half wide.
        F_sub = i_f_t @ in_f   # substrate
        F_sup = o_f_t @ o_f    # superstrate

        # Tangential fields for each incident mode (columns = modes):
        #   inc:   forward modes in substrate, no backward
        #   refl:  backward modes in substrate via S11
        #   trans: forward modes in superstrate via S21
        inc_fields = F_sub[:, :half]            # (4*num_h, half)
        refl_fields = F_sub[:, half:] @ S11     # (4*num_h, half)
        trans_fields = F_sup[:, :half] @ S21    # (4*num_h, half)

        # Extract E-field rows: Ey and Ex
        Ey = slice(2 * num_h, 3 * num_h)
        Ex = slice(3 * num_h, 4 * num_h)

        inc_E = jnp.concatenate([inc_fields[Ey], inc_fields[Ex]], axis=0)
        refl_E = jnp.concatenate([refl_fields[Ey], refl_fields[Ex]], axis=0)
        trans_E = jnp.concatenate([trans_fields[Ey], trans_fields[Ex]], axis=0)

        # Per-mode normalisation scalar: Ey diagonal for TE, Ex diagonal for TM
        te_norm = jnp.diag(inc_fields[Ey][:, :num_h])        # Ey(h) for TE mode h
        tm_norm = jnp.diag(inc_fields[Ex][:, num_h:])         # Ex(h) for TM mode h
        norm = jnp.concatenate([te_norm, tm_norm])             # (half,)

        r = refl_E / norm[None, :]
        t = trans_E / norm[None, :]
        return r, t
