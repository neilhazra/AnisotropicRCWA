from __future__ import annotations

from dataclasses import dataclass

from .backend import jnp

from .solver import Solver
from .stack import Stack

DEFAULT_NUM_POINTS_X = 512
DEFAULT_NUM_POINTS_Z = 65
DEFAULT_NUM_POINTS_RCWA = 2048
SUPPORTED_COMPONENTS = ["-H_y", "H_x", "E_y", "E_x"]


Sub2LayBlocks = tuple[jnp.ndarray, jnp.ndarray]
LayerModes = tuple[jnp.ndarray, jnp.ndarray]


@dataclass
class _Sub2LayData:
    substrate_faces: list[Sub2LayBlocks]


@dataclass
class _Lay2SupData:
    superstrate_faces: list[jnp.ndarray]


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[Visualize] {message}")


def _identity_sub2lay_blocks(num_modes: int, dtype=jnp.complex128) -> Sub2LayBlocks:
    return (
        jnp.eye(num_modes, dtype=dtype),
        jnp.zeros((num_modes, num_modes), dtype=dtype),
    )


def _propagation_diagonals(
    eigenvalues: jnp.ndarray,
    thickness_norm: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    half = eigenvalues.shape[0] // 2
    forward = jnp.exp(eigenvalues[:half] * thickness_norm)
    backward = jnp.exp(-eigenvalues[half:] * thickness_norm)
    return forward, backward


def _sub2lay_after_block(sub2lay_blocks: Sub2LayBlocks, block) -> Sub2LayBlocks:
    A21, A22 = sub2lay_blocks
    B11, B12, B21, B22 = block
    system = jnp.eye(A22.shape[0], dtype=A22.dtype) - A22 @ B11
    rhs = jnp.concatenate([A21, A22 @ B12], axis=1)
    solution = jnp.linalg.solve(system, rhs)
    inv_a_A21 = solution[:, : A21.shape[1]]
    inv_a_A22_B12 = solution[:, A21.shape[1] :]
    return (
        B21 @ inv_a_A21,
        B22 + B21 @ inv_a_A22_B12,
    )


def _sub2lay_after_propagation(
    sub2lay_blocks: Sub2LayBlocks,
    eigenvalues: jnp.ndarray,
    thickness_norm: float,
) -> Sub2LayBlocks:
    A21, A22 = sub2lay_blocks
    forward, backward = _propagation_diagonals(eigenvalues, thickness_norm)
    return (
        forward[:, None] * A21,
        (forward[:, None] * A22) * backward[None, :],
    )


def _lay2sup_before_block(block, lay2sup_B11: jnp.ndarray) -> jnp.ndarray:
    A11, A12, A21, A22 = block
    system = jnp.eye(A22.shape[0], dtype=A22.dtype) - A22 @ lay2sup_B11
    inv_a_A21 = jnp.linalg.solve(system, A21)
    return A11 + A12 @ (lay2sup_B11 @ inv_a_A21)


def _lay2sup_before_propagation(
    lay2sup_B11: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    thickness_norm: float,
) -> jnp.ndarray:
    forward, backward = _propagation_diagonals(eigenvalues, thickness_norm)
    return (backward[:, None] * lay2sup_B11) * forward[None, :]


def _build_layer_modes_and_interfaces(
    stack: Stack,
    N: int,
    num_points_rcwa: int,
    verbose: bool = False,
) -> tuple[list, list[LayerModes], tuple]:
    num_h = Stack.num_harmonics(N)
    mat_size = 4 * num_h
    n_layers = len(stack.layers)
    _log(
        verbose,
        f"Solving stack for field visualization: N={N}, {n_layers} layer(s), matrix size {mat_size}x{mat_size}",
    )
    _log(
        verbose,
        f"Diagonalizing substrate halfspace (block diagonalization, {num_h} independent 4x4 blocks)",
    )
    substrate_modes = Solver.harmonic_to_component_major_rows(
        Solver.get_substrate_mode_to_field(stack, N, num_points_rcwa, verbose=verbose)
    )
    _log(
        verbose,
        f"Building substrate tangential transform (matrix multiply, {mat_size}x{mat_size})",
    )
    prev_mode_tangential = (
        stack.substrate_reduced_to_tangential_field_transform_component_major(N)
        @ substrate_modes
    )

    interface_blocks = []
    layer_modes = []
    _log(verbose, f"Building Q matrices for {n_layers} layer(s)")
    for i, layer in enumerate(stack.layers):
        _log(verbose, f"--- Layer {i + 1}/{n_layers} ---")
        q_layer = stack.layer_Q_matrix_normalized(
            i,
            N,
            num_points=num_points_rcwa,
            verbose=verbose,
        )
        layer_tangential = stack.layer_reduced_to_tangential_field_transform_component_major(
            i,
            N,
            num_points_rcwa,
        )
        eigenvalues, mode_fields = Solver.diagonalize_sort_layer_system(
            q_layer,
            verbose=verbose,
            is_homogeneous=layer.is_homogeneous,
        )
        _log(
            verbose,
            f"  Computing tangential mode fields (matrix multiply, {mat_size}x{mat_size})",
        )
        current_layer_mode_tangential = layer_tangential @ mode_fields
        _log(
            verbose,
            f"  Computing interface S-matrix (linear solve + transfer-to-scattering, {mat_size}x{mat_size})",
        )
        interface_blocks.append(
            Solver.transfer_to_scattering(
                jnp.linalg.solve(current_layer_mode_tangential, prev_mode_tangential)
            )
        )
        layer_modes.append((eigenvalues, mode_fields))
        prev_mode_tangential = current_layer_mode_tangential

    _log(
        verbose,
        f"Diagonalizing superstrate halfspace (block diagonalization, {num_h} independent 4x4 blocks)",
    )
    superstrate_modes = Solver.harmonic_to_component_major_rows(
        Solver.get_superstrate_mode_to_field(stack, N, num_points_rcwa, verbose=verbose)
    )
    right_tangential = (
        stack.superstrate_reduced_to_tangential_field_transform_component_major(N)
        @ superstrate_modes
    )
    _log(
        verbose,
        f"Computing final superstrate interface S-matrix (linear solve, {mat_size}x{mat_size})",
    )
    final_interface = Solver.transfer_to_scattering(
        jnp.linalg.solve(right_tangential, prev_mode_tangential)
    )
    return interface_blocks, layer_modes, final_interface


def _build_sub2lay(
    interface_blocks: list,
    layer_modes: list[LayerModes],
    stack: Stack,
    N: int,
    verbose: bool = False,
) -> _Sub2LayData:
    half = 2 * Stack.num_harmonics(N)
    if interface_blocks:
        dtype = interface_blocks[0][0].dtype
    else:
        dtype = jnp.complex128

    _log(
        verbose,
        f"Building reduced sub2lay states ({len(interface_blocks)} stored layer substrate faces)",
    )
    sub2lay_blocks = _identity_sub2lay_blocks(half, dtype=dtype)
    substrate_faces: list[Sub2LayBlocks] = []
    for i, interface_block in enumerate(interface_blocks):
        sub2lay_blocks = _sub2lay_after_block(sub2lay_blocks, interface_block)
        substrate_faces.append(sub2lay_blocks)
        eigenvalues, _ = layer_modes[i]
        sub2lay_blocks = _sub2lay_after_propagation(
            sub2lay_blocks,
            eigenvalues,
            stack.thickness_normalized(i),
        )
    return _Sub2LayData(substrate_faces=substrate_faces)


def _build_lay2sup(
    interface_blocks: list,
    layer_modes: list[LayerModes],
    final_interface,
    stack: Stack,
    verbose: bool = False,
) -> _Lay2SupData:
    _log(
        verbose,
        f"Building reduced lay2sup states ({len(layer_modes)} stored layer superstrate faces)",
    )
    superstrate_faces: list[jnp.ndarray] = [None] * len(layer_modes)
    lay2sup_B11 = final_interface[0]
    for i in range(len(layer_modes) - 1, -1, -1):
        superstrate_faces[i] = lay2sup_B11
        eigenvalues, _ = layer_modes[i]
        lay2sup_B11 = _lay2sup_before_propagation(
            lay2sup_B11,
            eigenvalues,
            stack.thickness_normalized(i),
        )
        if i > 0:
            lay2sup_B11 = _lay2sup_before_block(interface_blocks[i], lay2sup_B11)
    return _Lay2SupData(superstrate_faces=superstrate_faces)


def _solve_stack(
    stack: Stack,
    N: int,
    num_points_rcwa: int,
    verbose: bool = False,
) -> tuple[_Sub2LayData, _Lay2SupData, list[LayerModes]]:
    interface_blocks, layer_modes, final_interface = _build_layer_modes_and_interfaces(
        stack,
        N,
        num_points_rcwa,
        verbose=verbose,
    )
    sub2lay = _build_sub2lay(interface_blocks, layer_modes, stack, N, verbose=verbose)
    lay2sup = _build_lay2sup(
        interface_blocks,
        layer_modes,
        final_interface,
        stack,
        verbose=verbose,
    )
    _log(verbose, "Stack solve complete")
    return sub2lay, lay2sup, layer_modes


def _layer_face_coefficients(
    sub2lay: _Sub2LayData,
    lay2sup: _Lay2SupData,
    layer_modes: list[LayerModes],
    stack: Stack,
    N: int,
    incident_pol: str,
    layer_index: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    incident_coeffs = jnp.zeros(2 * Stack.num_harmonics(N), dtype=jnp.complex128)
    incident_coeffs[Solver.zero_order_mode_index(N, incident_pol.upper())] = 1.0

    substrate_side_A21, substrate_side_A22 = sub2lay.substrate_faces[layer_index]
    superstrate_side_B11 = lay2sup.superstrate_faces[layer_index]
    eigenvalues, _ = layer_modes[layer_index]
    thickness_norm = stack.thickness_normalized(layer_index)
    substrate_side_B11 = _lay2sup_before_propagation(
        superstrate_side_B11,
        eigenvalues,
        thickness_norm,
    )
    substrate_side_forward = jnp.linalg.solve(
        jnp.eye(substrate_side_A22.shape[0], dtype=substrate_side_A22.dtype)
        - substrate_side_A22 @ substrate_side_B11,
        substrate_side_A21 @ incident_coeffs,
    )
    substrate_side_coeffs = jnp.concatenate(
        [substrate_side_forward, substrate_side_B11 @ substrate_side_forward]
    )

    superstrate_side_A21, superstrate_side_A22 = _sub2lay_after_propagation(
        (substrate_side_A21, substrate_side_A22),
        eigenvalues,
        thickness_norm,
    )
    superstrate_side_forward = jnp.linalg.solve(
        jnp.eye(superstrate_side_A22.shape[0], dtype=superstrate_side_A22.dtype)
        - superstrate_side_A22 @ superstrate_side_B11,
        superstrate_side_A21 @ incident_coeffs,
    )
    superstrate_side_coeffs = jnp.concatenate(
        [superstrate_side_forward, superstrate_side_B11 @ superstrate_side_forward]
    )
    return substrate_side_coeffs, superstrate_side_coeffs


# return component major fields at a current z distance inside layer measured from substrate side
def _field_k_at_batched_depths(
    stack: Stack,
    layer_index: int,
    layer_modes,
    substrateside_coeffs: jnp.ndarray,
    superstrateside_coeffs: jnp.ndarray,
    z_nms: float,
) -> jnp.ndarray:
    layer = stack.layers[layer_index]
    z_norm = 2 * jnp.pi * jnp.clip(z_nms, 0.0, layer.thickness_nm) / stack.wavelength_nm
    thickness_norm = jnp.array(stack.thickness_normalized(layer_index))
    eigenvalues, mode_fields = layer_modes[layer_index]
    half = eigenvalues.shape[0] // 2
    modal_coeffs = jnp.concatenate(
        [
            jnp.exp(eigenvalues[:half][None, :] * z_norm[:, None])
            * substrateside_coeffs[:half][None, :],
            jnp.exp(
                -eigenvalues[half:][None, :]
                * (thickness_norm[None, None] - z_norm[:, None])
            )
            * superstrateside_coeffs[half:][None, :],
        ],
        axis=-1,
    )
    field = mode_fields[None, ...] @ modal_coeffs[..., None]
    return jnp.asarray(field, dtype=jnp.complex128)[..., 0]


def _create_layer_profile(
    stack: Stack,
    sub2lay: _Sub2LayData,
    lay2sup: _Lay2SupData,
    layer_modes: list[LayerModes],
    N: int,
    layer_index: int,
    incident_pol: str,
    num_points_x: int,
    num_points_z: int,
    num_points_rcwa: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    substrate_side_coeffs, superstrate_side_coeffs = _layer_face_coefficients(
        sub2lay,
        lay2sup,
        layer_modes,
        stack,
        N,
        incident_pol=incident_pol,
        layer_index=layer_index,
    )
    layer = stack.layers[layer_index]
    x_nm = jnp.asarray(layer.sample_points(num_points_x))
    z_nm = jnp.linspace(0.0, layer.thickness_nm, num_points_z)
    fields = _field_k_at_batched_depths(
        stack,
        layer_index,
        layer_modes,
        substrate_side_coeffs,
        superstrate_side_coeffs,
        z_nm,
    )
    tangential_transform = stack.layer_reduced_to_tangential_field_transform_component_major(
        layer_index,
        N,
        num_points_rcwa,
    )
    tangential_field = (tangential_transform[None, ...] @ fields[..., None])[..., 0]
    num_h = stack.num_harmonics(N)
    __Hy = tangential_field[:, 0 * num_h : 1 * num_h]
    _Hx = tangential_field[:, 1 * num_h : 2 * num_h]
    _Ey = tangential_field[:, 2 * num_h : 3 * num_h]
    _Ex = tangential_field[:, 3 * num_h : 4 * num_h]
    fields_array = jnp.stack([__Hy, _Hx, _Ey, _Ex], axis=0)
    num_h = fields_array.shape[-1]
    x_min_nm, x_max_nm = layer.x_domain_nm
    period_nm = x_max_nm - x_min_nm
    orders = Stack.harmonic_orders(N)
    phase = (
        1j
        * (stack.kappa_inv_nm + 2 * jnp.pi * orders[None, :] / period_nm)
        * (x_nm[:, None] - x_min_nm)
    )
    field_xz = jnp.einsum("xn, fzn->fzx", jnp.exp(phase), fields_array)
    return x_nm, z_nm, field_xz


def _iter_layer_xz_profiles(
    stack: Stack,
    sub2lay: _Sub2LayData,
    lay2sup: _Lay2SupData,
    layer_modes: list[LayerModes],
    N: int,
    num_points_x: int,
    num_points_z: int,
    num_points_rcwa: int,
    verbose: bool = False,
):
    for layer_index in range(len(stack.layers)):
        _log(
            verbose,
            f"Reconstructing fields for layer {layer_index + 1}/{len(stack.layers)} (TE + TM)",
        )
        x_nm, z_nm, field_te_czx = _create_layer_profile(
            stack,
            sub2lay,
            lay2sup,
            layer_modes,
            N,
            layer_index,
            "TE",
            num_points_x,
            num_points_z,
            num_points_rcwa,
        )
        x_tm_nm, z_tm_nm, field_tm_czx = _create_layer_profile(
            stack,
            sub2lay,
            lay2sup,
            layer_modes,
            N,
            layer_index,
            "TM",
            num_points_x,
            num_points_z,
            num_points_rcwa,
        )
        if not jnp.allclose(x_nm, x_tm_nm, atol=1e-12, rtol=1e-12):
            raise ValueError("TE and TM visualizations must share the same x grid.")
        if not jnp.allclose(z_nm, z_tm_nm, atol=1e-12, rtol=1e-12):
            raise ValueError("TE and TM visualizations must share the same z grid.")
        yield x_nm, z_nm, field_te_czx, field_tm_czx


def create_layer_xz_profiles(
    stack: Stack,
    N: int,
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_z: int = DEFAULT_NUM_POINTS_Z,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    verbose: bool = False,
) -> list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    if verbose:
        print(
            f"[Visualize] Creating layer xz profiles: N={N}, {len(stack.layers)} layer(s), {num_points_x}x pts, {num_points_z}z pts"
        )
    sub2lay, lay2sup, layer_modes = _solve_stack(
        stack,
        N,
        num_points_rcwa,
        verbose=verbose,
    )
    return list(
        _iter_layer_xz_profiles(
            stack,
            sub2lay,
            lay2sup,
            layer_modes,
            N,
            num_points_x,
            num_points_z,
            num_points_rcwa,
            verbose=verbose,
        )
    )


def create_stack_xz_profile(
    stack: Stack,
    N: int,
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_z: int = DEFAULT_NUM_POINTS_Z,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    verbose: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if verbose:
        print(
            f"[Visualize] Creating layer xz profiles: N={N}, {len(stack.layers)} layer(s), {num_points_x}x pts, {num_points_z}z pts"
        )
    sub2lay, lay2sup, layer_modes = _solve_stack(
        stack,
        N,
        num_points_rcwa,
        verbose=verbose,
    )
    if not stack.layers:
        raise ValueError("Stack must contain at least one layer.")

    x_nm = None
    z_segments_nm = []
    field_te_segments_czx = []
    field_tm_segments_czx = []
    z_offset_nm = 0.0
    for layer_index, (
        layer_x_nm,
        layer_z_nm,
        layer_field_te_xz,
        layer_field_tm_xz,
    ) in enumerate(
        _iter_layer_xz_profiles(
            stack,
            sub2lay,
            lay2sup,
            layer_modes,
            N,
            num_points_x,
            num_points_z,
            num_points_rcwa,
            verbose=verbose,
        )
    ):
        layer_x_nm = jnp.asarray(layer_x_nm, dtype=jnp.float64)
        layer_z_nm = jnp.asarray(layer_z_nm, dtype=jnp.float64) + z_offset_nm
        layer_field_te_xz = jnp.asarray(layer_field_te_xz, dtype=jnp.complex128)
        layer_field_tm_xz = jnp.asarray(layer_field_tm_xz, dtype=jnp.complex128)
        if x_nm is None:
            x_nm = layer_x_nm
        elif not jnp.allclose(x_nm, layer_x_nm, atol=1e-12, rtol=1e-12):
            raise ValueError("All stitched RCWA layers must share the same x grid.")
        if layer_index > 0:
            layer_z_nm = layer_z_nm[1:]
            layer_field_te_xz = layer_field_te_xz[:, 1:, :]
            layer_field_tm_xz = layer_field_tm_xz[:, 1:, :]
        z_segments_nm.append(layer_z_nm)
        field_te_segments_czx.append(layer_field_te_xz)
        field_tm_segments_czx.append(layer_field_tm_xz)
        z_offset_nm += float(stack.layers[layer_index].thickness_nm)

    z_nm = jnp.concatenate(z_segments_nm, axis=0)
    field_te_czx = jnp.concatenate(field_te_segments_czx, axis=1)
    field_tm_czx = jnp.concatenate(field_tm_segments_czx, axis=1)
    return x_nm, z_nm, field_te_czx, field_tm_czx
