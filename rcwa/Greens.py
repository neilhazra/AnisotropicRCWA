from __future__ import annotations

import jax

import jax.numpy as jnp
from .solver import Solver
from .stack import Stack


def _layer_boundaries_nm(stack):
    thicknesses = jnp.asarray([layer.thickness_nm for layer in stack.layers])
    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(thicknesses)])


def _electric_current_discontinuity_xyz_homogeneous_n0_from_eps_zz(
    kappa_normalized,
    eps_zz,
    N: int,
    dtype,
) -> jnp.ndarray:
    assert N == 0
    num_h = Stack.num_harmonics(N)
    zero = Stack.zero_harmonic_index(N)
    jump = jnp.zeros((4 * num_h, 3), dtype=dtype)
    jump = jump.at[0 * num_h + zero, 0].set(1.0)
    jump = jump.at[1 * num_h + zero, 1].set(1.0)
    jump = jump.at[3 * num_h + zero, 2].set(kappa_normalized / eps_zz)
    return jump


def _diagonalize(args):
    layer_is_homogeneous, q_layer = args
    return jax.lax.cond(
        layer_is_homogeneous,
        Solver._diagonalize_sort_homogeneous_layer_system,
        Solver._diagonalize_sort_dense_layer_system,
        q_layer,
        False,
    )


def _interface(current_layer_mode_tangential, prev_mode_tangential):
    transfer = jnp.linalg.solve(
        current_layer_mode_tangential,
        prev_mode_tangential,
    )
    return Solver.transfer_to_scattering(transfer)


def build_blocks(stack: Stack, N: int, num_points_rcwa: int):
    substrate_modes = Solver.harmonic_to_component_major_rows(
        Solver.get_substrate_mode_to_field(stack, N, num_points_rcwa)
    )
    prev_mode_tangential = (
        stack.substrate_reduced_to_tangential_field_transform_component_major(N)
        @ substrate_modes
    )
    layer_tangentials = jnp.array(
        stack.build_all_layer_reduced_to_tangential_field_transform_component_major(
            N,
            num_points_rcwa,
        )
    )
    layers_homo = jnp.array([layer.is_homogeneous for layer in stack.layers])
    Qs = jnp.array(stack.build_all_Q_matrices_normalized(N, num_points_rcwa))
    normalized_thicknesses = jnp.array(
        [stack.thickness_normalized(i) for i, _ in enumerate(stack.layers)]
    )

    eigs, mode_fields = jax.lax.map(
        _diagonalize,
        (layers_homo, Qs),
        batch_size=16,
    )
    layer_mode_tangential = jax.vmap(jnp.matmul)(layer_tangentials, mode_fields)

    superstrate_modes = Solver.harmonic_to_component_major_rows(
        Solver.get_superstrate_mode_to_field(stack, N, num_points_rcwa)
    )
    right_tangential = (
        stack.superstrate_reduced_to_tangential_field_transform_component_major(N)
        @ superstrate_modes
    )

    lefts = jnp.concatenate(
        [prev_mode_tangential[None], layer_mode_tangential],
        axis=0,
    )
    rights = jnp.concatenate([layer_mode_tangential, right_tangential[None]], axis=0)
    block_interface = jax.vmap(_interface)(rights, lefts)
    block_propagation = jax.vmap(Solver.modal_propagation_scattering_matrix)(
        eigs,
        normalized_thicknesses,
    )
    return eigs, layer_mode_tangential, block_interface, block_propagation


def interleave(block_interface, block_propagation):
    """Interleave [I_0, P_0, I_1, P_1, ..., I_{n-1}, P_{n-1}, I_n].

    block_interface:    length n+1 along leading axis
    block_propagation:  length n   along leading axis
    returns:            length 2n+1
    """
    def _interleave_leaf(I, P):
        out = jnp.empty((2 * P.shape[0] + 1,) + I.shape[1:], dtype=I.dtype)
        out = out.at[0::2].set(I)
        out = out.at[1::2].set(P)
        return out

    return jax.tree.map(_interleave_leaf, block_interface, block_propagation)


def _star_identity(half, dtype):
    zero = jnp.zeros((half, half), dtype=dtype)
    eye = jnp.eye(half, dtype=dtype)
    return jnp.stack((zero, eye, eye, zero), axis=0)


def redheffer_star(A, B):
    """A, B: [..., 4, h, h] in order [S11, S12, S21, S22]."""
    A11, A12, A21, A22 = (A[..., i, :, :] for i in range(4))
    B11, B12, B21, B22 = (B[..., i, :, :] for i in range(4))
    h = A11.shape[-1]
    eye = jnp.eye(h, dtype=A.dtype)
    rhs = jnp.concatenate([A21, A22 @ B12], axis=-1)
    sol = jnp.linalg.solve(eye - A22 @ B11, rhs)
    iA21, iA22B12 = sol[..., :h], sol[..., h:]
    return jnp.stack(
        [
            A11 + A12 @ B11 @ iA21,
            A12 @ (B12 + B11 @ iA22B12),
            B21 @ iA21,
            B22 + B21 @ iA22B12,
        ],
        axis=-3,
    )


# cross[q, i, j] = redheffer_star(L[q, i], R[q, j])
_star_kij = jax.vmap(
    jax.vmap(
        jax.vmap(redheffer_star, (None, 0)),
        (0, None),
    ),
    (0, 0),
)


def _level(out, n, b):
    k = n // (2 * b)
    blocks = out.reshape(k, 2 * b, k, 2 * b, *out.shape[2:])
    q = jnp.arange(k)
    diag = blocks[q, :, q]
    left = diag[:, :b, b - 1]
    right = diag[:, b, b:]
    cross = _star_kij(left, right)
    return blocks.at[q, :b, q, b:].set(cross).reshape(n, n, *out.shape[2:])


def all_interval_stars(xs, identity):
    """Return all upper-triangle interval products of an associative star chain."""
    n = xs.shape[0]
    padded_n = 1 << (n - 1).bit_length() if n > 1 else 1
    if padded_n != n:
        pad = jnp.broadcast_to(identity, (padded_n - n, *identity.shape))
        xs = jnp.concatenate([xs, pad], axis=0)

    idx = jnp.arange(padded_n)
    out = jnp.zeros((padded_n, padded_n, *xs.shape[1:]), xs.dtype)
    out = out.at[idx, idx].set(xs)
    for ell in range(padded_n.bit_length() - 1):
        out = _level(out, padded_n, 1 << ell)
    return out if padded_n == n else out[:n, :n]


def build_compact_scattering_graph(stack: Stack, N: int, num_points_rcwa: int):
    eigs, layer_mode_tangential, block_interface, block_propagation = build_blocks(
        stack,
        N,
        num_points_rcwa,
    )
    half = block_interface[0].shape[-1]
    identity = _star_identity(half, block_interface[0].dtype)
    interfaces = jnp.stack(block_interface, axis=1)
    propagations = jnp.stack(block_propagation, axis=1)
    n_layers = propagations.shape[0]

    blocks = jnp.stack(interleave(block_interface, block_propagation), axis=1)
    sub2lay = jax.lax.associative_scan(redheffer_star, blocks, axis=0)[
        0 : 2 * n_layers : 2
    ]

    def reverse_redheffer_star(A, B):
        return redheffer_star(B, A)

    lay2sup = jax.lax.associative_scan(
        reverse_redheffer_star,
        blocks,
        axis=0,
        reverse=True,
    )[2 : 2 * n_layers + 1 : 2]
    layers = jnp.arange(n_layers)
    lower = layers[:, None]
    upper = layers[None, :]
    table_shape = (n_layers, n_layers, *identity.shape)

    adjacent_segments = jnp.broadcast_to(interfaces[upper], table_shape)
    identity_segments = jnp.broadcast_to(identity, table_shape)
    if n_layers == 1:
        up_segments = identity_segments
    else:
        interior_steps = jax.vmap(redheffer_star)(
            interfaces[1:-1],
            propagations[1:],
        )
        interior_segments = all_interval_stars(interior_steps, identity)
        segment_starts = jnp.minimum(lower, n_layers - 2)
        segment_stops = jnp.minimum(jnp.maximum(upper - 2, 0), n_layers - 2)
        interior_candidates = interior_segments[segment_starts, segment_stops]
        middle_candidates = jax.vmap(jax.vmap(redheffer_star))(
            interior_candidates,
            adjacent_segments,
        )
        up_segments = jnp.where(
            (upper <= lower)[:, :, None, None, None],
            identity_segments,
            jnp.where(
                (upper == lower + 1)[:, :, None, None, None],
                adjacent_segments,
                middle_candidates,
            ),
        )
    return eigs, layer_mode_tangential, sub2lay, lay2sup, up_segments


def _propagation(eigs_layer, z):
    """Stack a layer modal-propagation S-matrix into shape (4, h, h)."""
    return jnp.stack(
        Solver.modal_propagation_scattering_matrix(eigs_layer, z),
        axis=0,
    )


def _left_env_at(sub_to_layer, eigs_layer, z):
    """S-matrix from the substrate to depth ``z`` within a layer."""
    return redheffer_star(sub_to_layer, _propagation(eigs_layer, z))


def _right_env_at(layer_to_sup, eigs_layer, thickness, z):
    """S-matrix from depth ``z`` within a layer to the superstrate."""
    return redheffer_star(_propagation(eigs_layer, thickness - z), layer_to_sup)


def _up_between(
    up_segment,
    eigs_lower,
    thickness_lower,
    lower_z,
    eigs_upper,
    upper_z,
):
    """S-matrix from depth ``lower_z`` up to depth ``upper_z``."""
    return redheffer_star(
        redheffer_star(
            _propagation(eigs_lower, thickness_lower - lower_z),
            up_segment,
        ),
        _propagation(eigs_upper, upper_z),
    )


def _source_field_modes(kappa_normalized, mode2tangential_src, eps_zz_src, N):
    """Project the unit dipole field jump into source-layer modal coefficients."""
    field_discont = _electric_current_discontinuity_xyz_homogeneous_n0_from_eps_zz(
        kappa_normalized,
        eps_zz_src,
        N,
        mode2tangential_src.dtype,
    )
    return jnp.linalg.solve(mode2tangential_src, field_discont)


def _source_right_scattering_block(left_env_at_src, right_env_at_src):
    """Return the right-going source response block and matching identity."""
    A11, _, _, A22 = left_env_at_src
    B11, _, _, _ = right_env_at_src
    eye = jnp.eye(A11.shape[0], dtype=A11.dtype)
    I_AB_inv = jnp.linalg.solve(eye - A22 @ B11, eye)

    source_right = jnp.block(
        [
            [I_AB_inv, -I_AB_inv @ A22],
            [B11 @ I_AB_inv, -B11 @ I_AB_inv @ A22],
        ]
    )
    return source_right, eye


def _right_observation_M(C_segment, R_obs_right, source_right, eye):
    _, _, C21, C22 = C_segment

    forward = jnp.linalg.solve(eye - C22 @ R_obs_right, C21)
    reflected_forward = R_obs_right @ forward

    h = forward.shape[-1]
    source_right_top = source_right[:h, :]

    return jnp.concatenate(
        [
            forward @ source_right_top,
            reflected_forward @ source_right_top,
        ],
        axis=-2,
    )


def _apply_observer(tangential2xyz_obs, mode2tangential_obs, M, field_discont_modes):
    """Modal coefficients -> tangential fields -> xyz E-field response of one unit dipole."""
    return tangential2xyz_obs @ mode2tangential_obs @ M @ field_discont_modes


def make_dynamic_compact_upward_green_evaluator(
    stack: Stack,
    N: int,
    num_points_rcwa: int,
):
    """Compact upward-only evaluator; ``src_layer``/``obs_layer`` may be JAX tracers.

    Returns ``evaluate_upward(src_layer, src_z_nm, obs_layer, obs_z_nm)`` that
    requires the observer be at or above the source. Skipping the down branch
    keeps the JAXpr free of an outer direction ``jax.lax.cond`` and produces a
    noticeably faster trace/compile than the bidirectional reciprocal evaluator.
    """
    eigs, mode2tangential, sub2lay, lay2sup, up_segments = build_compact_scattering_graph(
        stack,
        N,
        num_points_rcwa,
    )
    n_layers = len(stack.layers)
    thicknesses = jnp.asarray(
        [stack.thickness_normalized(i) for i in range(n_layers)]
    )
    eps_zz = jnp.asarray(
        [stack.layers[i].eps(jnp.array([0.0]))[0, 2, 2] for i in range(n_layers)]
    )
    tangential2xyz = jnp.asarray(
        [
            stack.layer_tangential_to_E_xyz_transform_component_major(
                i,
                N,
                num_points_rcwa,
            )
            for i in range(n_layers)
        ]
    )
    z_scale = 2.0 * jnp.pi / stack.wavelength_nm

    def evaluate_upward(src_layer, src_z_nm, obs_layer, obs_z_nm):
        src_layer = jnp.asarray(src_layer, dtype=jnp.int32)
        obs_layer = jnp.asarray(obs_layer, dtype=jnp.int32)
        src_z = z_scale * src_z_nm
        obs_z = z_scale * obs_z_nm

        field_discont_modes = _source_field_modes(
            stack.kappa_normalized,
            mode2tangential[src_layer],
            eps_zz[src_layer],
            N,
        )
        A_src = _left_env_at(sub2lay[src_layer], eigs[src_layer], src_z)
        B_src = _right_env_at(
            lay2sup[src_layer],
            eigs[src_layer],
            thicknesses[src_layer],
            src_z,
        )
        source_right, eye = _source_right_scattering_block(A_src, B_src)

        R_obs_right = _right_env_at(
            lay2sup[obs_layer],
            eigs[obs_layer],
            thicknesses[obs_layer],
            obs_z,
        )[0]
        C = jax.lax.cond(
            src_layer == obs_layer,
            lambda _: _propagation(eigs[src_layer], obs_z - src_z),
            lambda _: _up_between(
                up_segments[src_layer, obs_layer],
                eigs[src_layer],
                thicknesses[src_layer],
                src_z,
                eigs[obs_layer],
                obs_z,
            ),
            operand=None,
        )
        M = _right_observation_M(C, R_obs_right, source_right, eye)
        return _apply_observer(
            tangential2xyz[obs_layer],
            mode2tangential[obs_layer],
            M,
            field_discont_modes,
        )

    return evaluate_upward


def make_reciprocal_compact_green_evaluator(
    stack: Stack,
    N: int,
    num_points_rcwa: int,
):
    """Bidirectional evaluator built on the upward evaluator via reciprocity.

    For ``src_z_abs <= obs_z_abs`` the upward evaluator is invoked directly.
    Otherwise we apply Fourier-domain reciprocity
    ``G(k; r, r') = P G(k; r', r)^T P`` where ``P = diag(-1, 1, 1)`` flips
    components that are odd under k_x -> -k_x.
    """
    upward = make_dynamic_compact_upward_green_evaluator(stack, N, num_points_rcwa)
    boundaries_nm = _layer_boundaries_nm(stack)
    kappa_parity = jnp.asarray([-1.0, 1.0, 1.0])

    def evaluate(src_layer, src_z_nm, obs_layer, obs_z_nm):
        src_layer = jnp.asarray(src_layer, dtype=jnp.int32)
        obs_layer = jnp.asarray(obs_layer, dtype=jnp.int32)
        src_abs = boundaries_nm[src_layer] + src_z_nm
        obs_abs = boundaries_nm[obs_layer] + obs_z_nm

        def reciprocal_downward(_):
            swapped = upward(obs_layer, obs_z_nm, src_layer, src_z_nm).T
            parity = kappa_parity.astype(swapped.dtype)
            return parity[:, None] * swapped * parity[None, :]

        return jax.lax.cond(
            src_abs <= obs_abs,
            lambda _: upward(src_layer, src_z_nm, obs_layer, obs_z_nm),
            reciprocal_downward,
            operand=None,
        )

    return evaluate


def evaluate_reciprocal_green_z_grid(
    build_stack,
    kappa,
    src_layer,
    z_src_nm,
    obs_layer,
    z_obs_nm,
    N,
    num_points_rcwa,
):
    """Evaluate a Cartesian source-z by observer-z Green tensor grid.

    Returns an array with shape ``[n_src_z, n_obs_z, 3, 3]``.
    """
    stack = build_stack(kappa)
    green = make_reciprocal_compact_green_evaluator(stack, N, num_points_rcwa)
    z_src_nm = jnp.asarray(z_src_nm).reshape(-1)
    z_obs_nm = jnp.asarray(z_obs_nm).reshape(-1)

    def evaluate_for_source(src_z):
        return jax.vmap(
            lambda obs_z: green(src_layer, src_z, obs_layer, obs_z)
        )(z_obs_nm)

    return jax.vmap(evaluate_for_source)(z_src_nm)


def map_reciprocal_green_z_grid_over_kappa(
    build_stack,
    kappas,
    src_layer,
    z_src_nm,
    obs_layer,
    z_obs_nm,
    N,
    num_points_rcwa,
    batch_size=None,
):
    """Map ``evaluate_reciprocal_green_z_grid`` over kappa values."""
    def evaluate_for_kappa(kappa):
        return evaluate_reciprocal_green_z_grid(
            build_stack,
            kappa,
            src_layer,
            z_src_nm,
            obs_layer,
            z_obs_nm,
            N,
            num_points_rcwa,
        )

    kappas = jnp.asarray(kappas)
    if batch_size is None:
        return jax.lax.map(evaluate_for_kappa, kappas)
    return jax.lax.map(evaluate_for_kappa, kappas, batch_size=batch_size)


def evaluate_reciprocal_green_observation_points(
    build_stack,
    kappa,
    src_layer,
    z_src_nm,
    obs_layers,
    z_obs_nm,
    N,
    num_points_rcwa,
):
    """Evaluate all source-z values against aligned observer layer/z points.

    Returns an array with shape ``[n_src_z, n_obs_points, 3, 3]``.
    """
    stack = build_stack(kappa)
    green = make_reciprocal_compact_green_evaluator(stack, N, num_points_rcwa)
    z_src_nm = jnp.asarray(z_src_nm).reshape(-1)
    obs_layers = jnp.asarray(obs_layers, dtype=jnp.int32).reshape(-1)
    z_obs_nm = jnp.asarray(z_obs_nm).reshape(-1)

    def evaluate_for_source(src_z):
        return jax.vmap(
            lambda obs_layer, obs_z: green(src_layer, src_z, obs_layer, obs_z)
        )(obs_layers, z_obs_nm)

    return jax.vmap(evaluate_for_source)(z_src_nm)


def map_reciprocal_green_observation_points_over_kappa(
    build_stack,
    kappas,
    src_layer,
    z_src_nm,
    obs_layers,
    z_obs_nm,
    N,
    num_points_rcwa,
    batch_size=None,
):
    """Map ``evaluate_reciprocal_green_observation_points`` over kappa values."""
    def evaluate_for_kappa(kappa):
        return evaluate_reciprocal_green_observation_points(
            build_stack,
            kappa,
            src_layer,
            z_src_nm,
            obs_layers,
            z_obs_nm,
            N,
            num_points_rcwa,
        )

    kappas = jnp.asarray(kappas)
    if batch_size is None:
        return jax.lax.map(evaluate_for_kappa, kappas)
    return jax.lax.map(evaluate_for_kappa, kappas, batch_size=batch_size)


def _green_components(response, k0):
    green = response / (1.0j * k0)
    return jnp.stack(
        [
            green[0, 0],
            green[1, 1],
            green[0, 2],
            green[2, 0],
            green[2, 2],
        ]
    )


def evaluate_reciprocal_green_observation_components(
    build_stack,
    kappa,
    src_layer,
    z_src_nm,
    obs_layers,
    z_obs_nm,
    N,
    num_points_rcwa,
):
    """Evaluate aligned observer points and return normalized Green components.

    Returns ``[G_xx, G_yy, G_xz, G_zx, G_zz]`` with shape
    ``[n_src_z, n_obs_points, 5]``.
    """
    stack = build_stack(kappa)
    green = make_reciprocal_compact_green_evaluator(stack, N, num_points_rcwa)
    k0 = 2.0 * jnp.pi / stack.wavelength_nm
    z_src_nm = jnp.asarray(z_src_nm).reshape(-1)
    obs_layers = jnp.asarray(obs_layers, dtype=jnp.int32).reshape(-1)
    z_obs_nm = jnp.asarray(z_obs_nm).reshape(-1)

    def evaluate_for_source(src_z):
        return jax.vmap(
            lambda obs_layer, obs_z: _green_components(
                green(src_layer, src_z, obs_layer, obs_z),
                k0,
            )
        )(obs_layers, z_obs_nm)

    return jax.vmap(evaluate_for_source)(z_src_nm)


def map_reciprocal_green_observation_components_over_kappa(
    build_stack,
    kappas,
    src_layer,
    z_src_nm,
    obs_layers,
    z_obs_nm,
    N,
    num_points_rcwa,
    batch_size=None,
):
    """Map component-valued observation-point Green evaluation over kappa."""
    def evaluate_for_kappa(kappa):
        return evaluate_reciprocal_green_observation_components(
            build_stack,
            kappa,
            src_layer,
            z_src_nm,
            obs_layers,
            z_obs_nm,
            N,
            num_points_rcwa,
        )

    kappas = jnp.asarray(kappas)
    if batch_size is None:
        return jax.lax.map(evaluate_for_kappa, kappas)
    return jax.lax.map(evaluate_for_kappa, kappas, batch_size=batch_size)



if __name__ == "__main__":
    import time

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import j0, j1, jv

    from .layer import Layer

    total_start = time.perf_counter()

    wavelength_nm = 1000.0

    eps_glass = 2.25  # n_glass = 1.5
    eps_air = 1.0

    # Many finite layers, but physically just glass below and air above.
    # This stresses the graph construction while matching a two-layer glass/air stack.
    num_layers = 13
    interface_layer_index = num_layers // 2 + 1
    layer_thickness_nm = 220.0
    layer_thicknesses_nm = np.full(num_layers, layer_thickness_nm)
    layer_eps = np.asarray([
        eps_glass if i < interface_layer_index else eps_air
        for i in range(num_layers)
    ])
    layer_boundaries_nm = np.concatenate([
        np.array([0.0]),
        np.cumsum(layer_thicknesses_nm),
    ])
    total_thickness_nm = float(layer_boundaries_nm[-1])
    src_layer_index = interface_layer_index - 1
    distance_below_interface_nm = 5.0
    src_local_z_nm = layer_thicknesses_nm[src_layer_index] - distance_below_interface_nm
    z_src_global_nm = layer_boundaries_nm[src_layer_index] + src_local_z_nm

    # k_parallel sweep along the solver's ky=0 direction. The stack is planar
    # and isotropic, so one solve per k_parallel is enough; the angular part of
    # the 2D inverse Fourier transform is handled analytically with Bessel
    # functions below.
    k0 = 2.0 * np.pi / wavelength_nm
    N_kappa = 1024
    kappa_max_over_k0 = 30.0
    kappa_max = kappa_max_over_k0 * k0
    dkappa = kappa_max / N_kappa
    kappas = (np.arange(N_kappa) + 0.5) * dkappa  # midpoint, avoids kappa=0
    hann_window = 0.5 * (1.0 + np.cos(np.pi * kappas / kappa_max))

    # Real-space (x, z) observation grid.
    Nx = 201
    x_max_nm = 2000.0
    x_grid_nm = np.linspace(-x_max_nm, x_max_nm, Nx)
    Nz = 240
    z_grid_nm = np.linspace(1.0, total_thickness_nm - 1.0, Nz)

    def build_stack(kappa_inv_nm: float) -> Stack:
        stack = Stack(
            wavelength_nm=wavelength_nm,
            kappa_inv_nm=kappa_inv_nm,
            eps_substrate=eps_glass,
            eps_superstrate=eps_air,
        )
        for thickness_nm, eps in zip(layer_thicknesses_nm, layer_eps):
            stack.add_layer(
                Layer.uniform(
                    thickness_nm=float(thickness_nm),
                    eps_tensor=float(eps) * jnp.eye(3, dtype=jnp.complex128),
                    x_domain_nm=(0.0, 1.0),
                )
            )
        return stack

    G_kz = np.zeros((N_kappa, Nz, 5), dtype=np.complex128)

    N = 0
    num_points_rcwa = 16

    for boundary_nm in layer_boundaries_nm[1:-1]:
        z_grid_nm[np.isclose(z_grid_nm, boundary_nm)] += 1e-6

    obs_layer_indices = np.searchsorted(
        layer_boundaries_nm[1:],
        z_grid_nm,
        side="right",
    )
    z_obs_local_nm = z_grid_nm - layer_boundaries_nm[obs_layer_indices]
    z_src_nm = jnp.asarray([src_local_z_nm])
    obs_layers = jnp.asarray(obs_layer_indices, dtype=jnp.int32)
    z_obs_nm = jnp.asarray(z_obs_local_nm)

    kappa_batch_size = 32
    print(
        f"Sweeping {N_kappa} kappa values x {Nz} z values "
        f"in JAX batches of {kappa_batch_size}..."
    )

    def solve_kappa_batch(kappa_values, src_layer, z_src, obs_layer_values, z_obs):
        return map_reciprocal_green_observation_components_over_kappa(
            build_stack,
            kappa_values,
            src_layer,
            z_src,
            obs_layer_values,
            z_obs,
            N,
            num_points_rcwa,
            batch_size=kappa_batch_size,
        )

    solve_kappa_batch = jax.jit(solve_kappa_batch)

    component_fields = solve_kappa_batch(
        jnp.asarray(kappas),
        jnp.asarray(src_layer_index, dtype=jnp.int32),
        z_src_nm,
        obs_layers,
        z_obs_nm,
    )[:, 0]
    all_fields = jax.device_get(component_fields)

    G_kz[:, :, :] = np.asarray(all_fields, dtype=np.complex128)
    print(f"  kappa step {N_kappa}/{N_kappa}")

    G_xx_kz = G_kz[:, :, 0]
    G_yy_kz = G_kz[:, :, 1]
    G_xz_kz = G_kz[:, :, 2]
    G_zx_kz = G_kz[:, :, 3]
    G_zz_kz = G_kz[:, :, 4]

    rho_grid_nm = np.abs(x_grid_nm)
    kr = kappas[:, None] * rho_grid_nm[None, :]
    J0 = j0(kr)
    J1 = j1(kr)
    J2 = jv(2, kr)
    radial_weight = (kappas * hann_window)[:, None]

    E_x_from_x_dipole = (
        dkappa
        / (4.0 * np.pi)
        * (
            ((radial_weight * (J0 - J2)).T @ G_xx_kz)
            + ((radial_weight * (J0 + J2)).T @ G_yy_kz)
        )
    )
    E_z_from_x_dipole = (
        dkappa
        / (2.0 * np.pi)
        * ((1.0j * radial_weight * J1).T @ G_zx_kz)
    )
    E_x_from_z_dipole = (
        dkappa
        / (2.0 * np.pi)
        * ((1.0j * radial_weight * J1).T @ G_xz_kz)
    )
    E_z_from_z_dipole = (
        dkappa
        / (2.0 * np.pi)
        * ((radial_weight * J0).T @ G_zz_kz)
    )

    x_dipole_field = np.sqrt(
        np.abs(E_x_from_x_dipole) ** 2 + np.abs(E_z_from_x_dipole) ** 2
    )
    z_dipole_field = np.sqrt(
        np.abs(E_x_from_z_dipole) ** 2 + np.abs(E_z_from_z_dipole) ** 2
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    def plot_panel(ax, data_xz, title):
        img_data = np.asarray(data_xz.T, dtype=np.float64)
        img_data = np.abs(img_data)
        source_distance_nm = np.sqrt(
            x_grid_nm[None, :] ** 2
            + (z_grid_nm[:, None] - z_src_global_nm) ** 2
        )
        near_source_radius_nm = 200.0
        near_source = source_distance_nm < near_source_radius_nm
        finite = np.isfinite(img_data) & (img_data > 0.0) & ~near_source
        print(f"{title}: finite pixels {np.count_nonzero(finite)}/{img_data.size}")
        if not np.any(finite):
            img_data = np.zeros_like(img_data)
            finite = np.ones_like(img_data, dtype=bool)
        finite_values = img_data[finite]
        vmin = np.percentile(finite_values, 1.0)
        vmax = np.percentile(finite_values, 99.0)
        if vmax <= vmin:
            vmin = 0.0
            vmax = np.max(finite_values)
        if vmax <= vmin:
            vmax = vmin + 1.0
        img_data = np.nan_to_num(img_data, nan=vmin, posinf=vmax, neginf=vmin)
        img_data = np.where(near_source, vmin, img_data)
        img_data = np.clip(img_data, vmin, vmax)
        img_data = (img_data - vmin) / (vmax - vmin)
        img = ax.imshow(
            img_data,
            extent=[x_grid_nm[0], x_grid_nm[-1], z_grid_nm[0], z_grid_nm[-1]],
            aspect="auto",
            origin="lower",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        for boundary_nm in layer_boundaries_nm[1:-1]:
            ax.axhline(boundary_nm, color="cyan", lw=0.45, alpha=0.55)
        ax.plot(
            0.0,
            z_src_global_nm,
            marker="*",
            color="white",
            ms=10,
            mec="black",
            mew=0.5,
        )
        ax.text(
            0.02,
            0.04,
            "glass",
            color="white",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )
        ax.text(
            0.02,
            0.96,
            "air",
            color="white",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("z (nm)")
        ax.set_title(title)
        plt.colorbar(
            img,
            ax=ax,
            label="linear amplitude contrast (1-99%, source masked)",
        )

    plot_panel(
        axes[0],
        x_dipole_field,
        r"$\sqrt{|E_x|^2+|E_z|^2}$  (x-oriented dipole)",
    )
    plot_panel(
        axes[1],
        z_dipole_field,
        r"$\sqrt{|E_x|^2+|E_z|^2}$  (z-oriented dipole)",
    )

    fig.suptitle(
        f"Dipole {distance_below_interface_nm:.0f} nm below glass-air interface  |  "
        f"{num_layers} finite layers  |  "
        f"lambda = {wavelength_nm:.0f} nm  |  cylindrical Hankel reconstruction"
    )
    plt.tight_layout()
    out_path = "alternating_stack_dipole_Gxx_Gzz.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    print(f"Total computation time: {time.perf_counter() - total_start:.2f} s")
    plt.show()
