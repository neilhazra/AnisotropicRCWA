from __future__ import annotations

from .backend import jnp

from .solver import Solver
from .stack import Stack

DEFAULT_NUM_POINTS_X = 512
DEFAULT_NUM_POINTS_Z = 65
DEFAULT_NUM_POINTS_RCWA = 2048
SUPPORTED_COMPONENTS = ["-H_y", "H_x", "E_y", "E_x"]


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[Visualize] {message}")


def _solve_stack(stack: Stack, N: int, num_points_rcwa: int, verbose: bool = False):
    #sub 2 sup is the scattering matrix that takes substrate modes
    # and propagates it to each interface
    # sup 2 sub takes the interface mode to superstrate modes, stiched together by tangential field continutity
    num_h = Stack.num_harmonics(N)
    mat_size = 4 * num_h
    n_layers = len(stack.layers)
    _log(verbose, f"Solving stack for field visualization: N={N}, {n_layers} layer(s), matrix size {mat_size}x{mat_size}")
    _log(verbose, f"Diagonalizing substrate halfspace (block diagonalization, {num_h} independent 4x4 blocks)")
    substrate_modes = Solver.harmonic_to_component_major_rows(
        Solver.get_substrate_mode_to_field(stack, N, num_points_rcwa, verbose=verbose)
    )
    _log(verbose, f"Building substrate tangential transform (matrix multiply, {mat_size}x{mat_size})")
    prev_mode_tangential = (
        stack.substrate_reduced_to_tangential_field_transform_component_major(N)
        @ substrate_modes
    )

    blocks = []
    layer_modes = []
    _log(verbose, f"Building Q matrices for {n_layers} layer(s)")
    for i, (layer, q_layer) in enumerate(
        zip(
            stack.layers,
            stack.build_all_Q_matrices_normalized(N, num_points_rcwa, verbose=verbose),
        )
    ):
        _log(verbose, f"--- Layer {i + 1}/{n_layers} ---")
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
        _log(verbose, f"  Computing tangential mode fields (matrix multiply, {mat_size}x{mat_size})")
        current_layer_mode_tangential = layer_tangential @ mode_fields
        _log(verbose, f"  Computing interface S-matrix (linear solve + transfer-to-scattering, {mat_size}x{mat_size})")
        blocks.append(
            Solver.transfer_to_scattering(
                jnp.linalg.solve(current_layer_mode_tangential, prev_mode_tangential) # this gives prior mode to current layer mode
            )
        )
        _log(verbose, f"  Computing modal propagation S-matrix")
        blocks.append(
            Solver.modal_propagation_scattering_matrix( # propagates mode through current layer
                eigenvalues,
                stack.thickness_normalized(i),
            )
        )
        layer_modes.append((eigenvalues, mode_fields))
        prev_mode_tangential = current_layer_mode_tangential

    _log(verbose, f"Diagonalizing superstrate halfspace (block diagonalization, {num_h} independent 4x4 blocks)")
    superstrate_modes = Solver.harmonic_to_component_major_rows(
        Solver.get_superstrate_mode_to_field(stack, N, num_points_rcwa, verbose=verbose)
    )
    right_tangential = (
        stack.superstrate_reduced_to_tangential_field_transform_component_major(N)
        @ superstrate_modes
    )
    _log(verbose, f"Computing final superstrate interface S-matrix (linear solve, {mat_size}x{mat_size})")
    blocks.append(
        Solver.transfer_to_scattering(jnp.linalg.solve(right_tangential, prev_mode_tangential))
    )

    half = blocks[0][0].shape[0]
    zero = jnp.zeros((half, half), dtype=blocks[0][0].dtype)
    eye = jnp.eye(half, dtype=blocks[0][0].dtype)
    identity = (zero, eye, eye, zero)

    n_blocks = len(blocks)
    _log(verbose, f"Building sub2lay prefix S-matrices ({n_blocks} Redheffer star products, forward pass)")
    sub2lay = [identity]
    for block in blocks:
        sub2lay.append(Solver.redheffer_star_product(sub2lay[-1], block))

    _log(verbose, f"Building lay2sup suffix S-matrices ({n_blocks} Redheffer star products, backward pass)")
    lay2sup = [identity for _ in range(len(blocks) + 1)]
    for i in range(len(blocks) - 1, -1, -1):
        lay2sup[i] = Solver.redheffer_star_product(blocks[i], lay2sup[i + 1])

    _log(verbose, "Stack solve complete")
    return sub2lay, lay2sup, layer_modes


def _layer_face_coefficients(
    sub2lay,
    lay2sup,
    N: int,
    incident_pol: str,
    layer_index: int,
) -> tuple[jnp.ndarray, jnp.ndarray]: # output in transfer matrix type arrangement (i.e all substrate side fields, all superstrate side fields)
    incident_coeffs = jnp.zeros(2 * Stack.num_harmonics(N), dtype=jnp.complex128)
    incident_coeffs[Solver.zero_order_mode_index(N, incident_pol.upper())] = 1.0
    substrate_side_i = 2 * layer_index + 1
    _, _, A21, A22 = sub2lay[substrate_side_i] # scattering matrix that takes substrate modes to the modes of substrate side of layer_i
                                    # note 0 index of prefix is the identity
    B11, _, _, _ = lay2sup[substrate_side_i] # scattering matrix that takes layer modes on substrate side of layer i into superstrate modes
    substrate_side_forward = jnp.linalg.solve(
        jnp.eye(A22.shape[0], dtype=A22.dtype) - A22 @ B11,
        A21 @ incident_coeffs,
    )
    substrate_side_coeffs = jnp.concatenate([substrate_side_forward, B11 @ substrate_side_forward])
    superstrate_side_i = substrate_side_i + 1
    _, _, A21, A22 = sub2lay[superstrate_side_i]
    B11, _, _, _ = lay2sup[superstrate_side_i]
    superstrate_side_forward = jnp.linalg.solve(
        jnp.eye(A22.shape[0], dtype=A22.dtype) - A22 @ B11,
        A21 @ incident_coeffs,
    )
    superstrate_side_coeffs = jnp.concatenate([superstrate_side_forward, B11 @ superstrate_side_forward])
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
    z_norm = 2 * jnp.pi * jnp.clip(z_nms, 0.0, layer.thickness_nm) / stack.wavelength_nm # [num_z]
    thickness_norm = jnp.array(stack.thickness_normalized(layer_index))
    eigenvalues, mode_fields = layer_modes[layer_index]
    half = eigenvalues.shape[0] // 2
    # scattering matrix we know all ports, and eigvals. 
    # TM would propagate left field modes through the materials with exp(eigval)
    # S-matrix we can propagate the left incident through (which is ordered such that all exp decay)
    # propagating left outgoing through would blow up
    # so instead propagate right incoming backwards through the material 
    modal_coeffs = jnp.concatenate(
        [
            jnp.exp(eigenvalues[:half][None, :] * z_norm[:, None]) * substrateside_coeffs[:half][None, :],
            jnp.exp(-eigenvalues[half:][None, :] * (thickness_norm[None, None] - z_norm[:, None])) * superstrateside_coeffs[half:][None, :],
        ], axis = -1
    )
    field = mode_fields[None, ...] @ modal_coeffs[..., None] # force mat mul with the z batched
    return jnp.asarray(field, dtype=jnp.complex128)[..., 0] # shape zs, field


def _create_layer_profile(
    stack: Stack,
    sub2lay,
    lay2sup,
    layer_modes,
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
        N,
        incident_pol=incident_pol,
        layer_index=layer_index,
    )
    layer = stack.layers[layer_index]
    x_nm = jnp.asarray(layer.sample_points(num_points_x))
    z_nm = jnp.linspace(0.0, layer.thickness_nm, num_points_z)
    fields = _field_k_at_batched_depths(stack, layer_index, layer_modes, substrate_side_coeffs, superstrate_side_coeffs, z_nm)
    tangential_transform = stack.layer_reduced_to_tangential_field_transform_component_major(
        layer_index,
        N,
        num_points_rcwa,
    )
    tangential_field = (tangential_transform[None, ...] @ fields[..., None])[...,0] # shape(z_nm, component major ordered fields)
    num_h = stack.num_harmonics(N)
    __Hy = tangential_field[:,0*num_h:1*num_h]
    _Hx = tangential_field[:, 1*num_h:2*num_h]
    _Ey = tangential_field[:, 2*num_h:3*num_h]    
    _Ex = tangential_field[:, 3*num_h:4*num_h]
    fields_array = jnp.stack([__Hy, _Hx, _Ey, _Ex], axis = 0) # 4 components, z_nm, -N to N harmonics
    num_h = fields_array.shape[-1]
    x_min_nm, x_max_nm = layer.x_domain_nm
    period_nm = x_max_nm - x_min_nm
    orders = Stack.harmonic_orders(N)
    phase = (
        1j
        * (stack.kappa_inv_nm + 2 * jnp.pi * orders[None, :] / period_nm)
        * (x_nm[:, None] - x_min_nm)
    )
    field_xz = jnp.einsum('xn, fzn->fzx', jnp.exp(phase), fields_array) # x harmonics, fields(4) zaxis harmonics -> fields x, z 
    return x_nm, z_nm, field_xz


def create_layer_xz_profiles(
    stack: Stack,
    N: int,
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_z: int = DEFAULT_NUM_POINTS_Z,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    verbose: bool = False,
) -> list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    if verbose:
        print(f"[Visualize] Creating layer xz profiles: N={N}, {len(stack.layers)} layer(s), {num_points_x}x pts, {num_points_z}z pts")
    sub2lay, lay2sup, layer_modes = _solve_stack(stack, N, num_points_rcwa, verbose=verbose)
    profiles = []
    for layer_index in range(len(stack.layers)):
        _log(verbose, f"Reconstructing fields for layer {layer_index + 1}/{len(stack.layers)} (TE + TM)")

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
        profiles.append((x_nm, z_nm, field_te_czx, field_tm_czx))
    return profiles


def create_stack_xz_profile(
    stack: Stack,
    N: int,
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_z: int = DEFAULT_NUM_POINTS_Z,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    verbose: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    profiles = create_layer_xz_profiles(
        stack,
        N,
        num_points_x=num_points_x,
        num_points_z=num_points_z,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    if not profiles:
        raise ValueError("Stack must contain at least one layer.")

    x_nm = None
    z_segments_nm = []
    field_te_segments_czx = []
    field_tm_segments_czx = []
    z_offset_nm = 0.0
    for layer_index, (layer_x_nm, layer_z_nm, layer_field_te_xz, layer_field_tm_xz) in enumerate(profiles):
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
