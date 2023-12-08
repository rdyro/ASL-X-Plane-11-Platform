from copy import copy

from jaxfi import jaxm


def bmv(A, x):
    """Batched matrix-vector product."""
    return (A @ x[..., None])[..., 0]


####################################################################################################
# partially linear, data-identified dynamics #######################################################
####################################################################################################


def dynamics(state, control, params):
    """Simple dynamics of an airplane."""
    x, y, z, v, vh, pitch, roll, yaw, dpitch, droll, dyaw = state

    # position
    xp = (v * jaxm.cos(yaw + 0 * params["heading_correction"])).reshape(())
    yp = (v * jaxm.sin(yaw + 0 * params["heading_correction"])).reshape(())
    dt = params["dt_sqrt"] ** 2

    dynamic_states = jaxm.stack([v, vh, pitch, roll, dpitch, droll, dyaw])
    statep_partial = bmv(params["Wx"], dynamic_states) + bmv(params["Wu"], control) + params["b"]

    statep = jaxm.cat([jaxm.stack([xp, yp]), statep_partial])
    return state + dt * statep


params0 = {
    "dt_sqrt": jaxm.sqrt(jaxm.array([0.5])),
    "heading_correction": jaxm.array([0.0]),
    "Wx": jaxm.randn((9, 7)),
    "Wu": jaxm.randn((9, 4)),
    "b": jaxm.randn(9),
}


@jaxm.jit
def fwd_fn(state, control, params):
    return jaxm.vmap(dynamics, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def f_fx_fu_fn(X, U, params):
    bshape = X.shape[:-1]
    X, U = X.reshape((-1, X.shape[-1])), U.reshape((-1, U.shape[-1]))
    f = fwd_fn(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    fsh, fxsh, fush = bshape + f.shape[-1:], bshape + fx.shape[-2:], bshape + fu.shape[-2:]
    return f.reshape(fsh), fx.reshape(fxsh), fu.reshape(fush)


####################################################################################################
# dynamics with integrated error ###################################################################
####################################################################################################


def int_dynamics(state, control, params):
    dt = params["dt_sqrt"] ** 2
    aero_state = state[:11]
    next_aero_state = dynamics(aero_state, control, params)
    pos_int = state[11:14]
    ang_int = state[14:17]
    pos_int = pos_int + dt * (next_aero_state[:3] - params["pos_ref"])
    ang_int = ang_int + dt * (next_aero_state[5:8] - params["ang_ref"])
    return jaxm.cat([next_aero_state, pos_int, ang_int])


def int_fwd_fn2(state, control, params):
    return jaxm.vmap(int_dynamics, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def int_f_fx_fu_fn(X, U, params):
    bshape = X.shape[:-1]
    X, U = X.reshape((-1, X.shape[-1])), U.reshape((-1, U.shape[-1]))
    f = int_fwd_fn2(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: int_fwd_fn2(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: int_fwd_fn2(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    fsh, fxsh, fush = bshape + f.shape[-1:], bshape + fx.shape[-2:], bshape + fu.shape[-2:]
    return f.reshape(fsh), fx.reshape(fxsh), fu.reshape(fush)


####################################################################################################
# NN dynamics ######################################################################################
####################################################################################################

nn_params0 = {
    "Wx0": jaxm.randn((32, 7)),
    "Wu0": jaxm.randn((32, 4)),
    "b0": jaxm.randn(32),
    "Wx1": jaxm.randn((32, 32)),
    "Wu1": jaxm.randn((32, 4)),
    "b1": jaxm.randn(32),
    "Wx2": jaxm.randn((9, 32)),
    "Wu2": jaxm.randn((9, 4)),
    "b2": jaxm.randn(9),
    "heading_correction": jaxm.array([0.0]),
    "dt_sqrt": jaxm.sqrt(jaxm.array([0.5])),
}


def nn_dynamics(state, control, params):
    """Simple dynamics of an airplane."""
    x, y, z, v, vh, pitch, roll, yaw, dpitch, droll, dyaw = state

    # position
    xp = (v * jaxm.cos(yaw + params["heading_correction"])).reshape(())
    yp = (v * jaxm.sin(yaw + params["heading_correction"])).reshape(())
    dt = params["dt_sqrt"] ** 2

    dt = params["dt_sqrt"] ** 2

    dynamic_states = jaxm.stack([v, vh, pitch, roll, dpitch, droll, dyaw])

    Z = dynamic_states
    for i in range(3):
        Z = params[f"Wx{i}"] @ Z + params[f"b{i}"] + params[f"Wu{i}"] @ control
        if i < 3 - 1:
            Z = jaxm.tanh(Z)
    return state + dt * jaxm.cat([jaxm.stack([xp, yp]), Z])


@jaxm.jit
def nn_fwd_fn(state, control, params):
    return jaxm.vmap(nn_dynamics, in_axes=(0, 0, None))(state, control, params)


@jaxm.jit
def nn_f_fx_fu_fn(X, U, params):
    if X.ndim == 3:
        return jaxm.vmap(nn_f_fx_fu_fn, in_axes=(0, 0, None))(X, U, params)
    f = nn_fwd_fn(X, U, params)
    fx = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: nn_fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=0
        )(x, u)
    )(X, U)
    fu = jaxm.vmap(
        lambda x, u: jaxm.jacobian(
            lambda x, u: nn_fwd_fn(x[None, ...], u[None, ...], params)[0, ...], argnums=1
        )(x, u)
    )(X, U)
    return f, fx, fu


####################################################################################################
