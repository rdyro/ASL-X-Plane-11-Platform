from __future__ import annotations

from typing import Callable
from pathlib import Path
import json
from copy import copy

import numpy as np
import torch
from torch import nn, Tensor
from torchvision import transforms as T
from torchvision.models import resnet50
from jaxfi import jaxm

from ..flight_control import dynamics

####################################################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = T.Compose(
    [
        T.Lambda(lambda x: torch.as_tensor(x).cuda()),
        T.Lambda(lambda x: x.transpose(-3, -1)),
        T.RandomRotation(10),
        T.ColorJitter(1.0, 1.0, 1.0, 0.5),
        T.Resize((640, 480)),
    ]
)

transform_eval = T.Compose(
    [
        T.Lambda(lambda x: torch.as_tensor(x).cuda()),
        T.Lambda(lambda x: x.transpose(-3, -1)),
        T.Resize((640, 480)),
    ]
)

####################################################################################################

default_model_path = Path(__file__).absolute().parents[1] / "data" / "resnet50_model.pt"


def get_model(
    model_path: Path | str | None = None,
    pretrained: bool = True,
    device: torch.device | None = None,
):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.weight.shape[-1], 3)
    model.conv1 = nn.Sequential(nn.BatchNorm2d(3), model.conv1)
    model.to(DEVICE if device is None else device)
    if pretrained:
        model_path = default_model_path if model_path is None else Path(model_path)
        model.load_state_dict(torch.load(model_path))
    model.eval()
    model = torch.compile(model)
    return model


def forward(model: nn.Module, frame_or_frames: Tensor):
    example_param = next(model.parameters())
    assert frame_or_frames.ndim in (3, 4)
    if frame_or_frames.ndim == 3:
        is_single_frame = True
        frames = frame_or_frames[None, ...]
    else:
        is_single_frame = False
        frames = frame_or_frames
    scale_factor = torch.tensor([1e3, 1e3, 1e2]).to(example_param)

    with torch.no_grad():
        Yp = model(transform_eval(frames)) * scale_factor
    return Yp[0] if is_single_frame else Yp


####################################################################################################

H_obs_default = torch.cat([torch.diag(torch.ones(3)), torch.zeros((11 - 3, 3))], -2).mT
P_default = torch.eye(11)
R_default = torch.diag(torch.as_tensor([61.0, 109.0, 6.26]))


def kalman_filter(
    f_fx_fu_fn: Callable,
    z_obs: Tensor,
    x0: Tensor,
    u0: Tensor,
    R: Tensor | None = None,
    H_obs: Tensor | None = None,
    P: Tensor | None = None,
):
    z_obs = z_obs.to(torch.float64)
    x0 = x0.to(torch.float64)
    u0 = u0.to(torch.float64)
    P = P_default.to(x0) if P is None else P.to(x0)
    R = R_default.to(x0) if R is None else R.to(x0)
    H_obs = H_obs_default.to(x0) if H_obs is None else H_obs.to(x0)
    f, fx, fu = [
        torch.from_numpy(np.array(x)).to(x0)
        for x in f_fx_fu_fn(
            jaxm.array(x0.cpu().detach().numpy()), jaxm.array(u0.cpu().detach().numpy())
        )
    ]

    # Kalman filter for (xp = f + fx x + fu u)
    xp = f
    P = fx @ P @ fx.mT + 1e-5 * torch.eye(x0.shape[-1]).to(x0)

    S = H_obs @ P @ H_obs.mT + R
    # K = P @ H_obs.mT @ torch.linalg.pinv(S)

    xp_estimate = xp + P @ H_obs.mT @ torch.linalg.solve(S, z_obs - H_obs @ xp)
    P_estimate = (torch.eye(x0.shape[-1]).to(x0) - P @ H_obs.mT @ torch.linalg.solve(S, H_obs)) @ P

    return xp_estimate, P_estimate


####################################################################################################


def get_dynamics(dynamics_path: str | Path | None = None):
    # bmv = lambda A, x: (A @ x[..., None])[..., 0]
    # default_dynamics_path = Path(__file__).absolute().parents[1] / "data" / "dynamics.json"

    # def _aero_dynamics(state, control, params):
    #    """Simple dynamics of an airplane."""
    #    x, y, z, v, vh, pitch, roll, yaw, dpitch, droll, dyaw = state

    #    # position
    #    xp = (v * jaxm.cos(yaw + 0 * params["heading_correction"])).reshape(())
    #    yp = (v * jaxm.sin(yaw + 0 * params["heading_correction"])).reshape(())
    #    dt = params["dt_sqrt"] ** 2

    #    dynamic_states = jaxm.stack([v, vh, pitch, roll, dpitch, droll, dyaw])
    #    statep_partial = (
    #        bmv(params["Wx"], dynamic_states) + bmv(params["Wu"], control) + params["b"]
    #    )

    #    statep = jaxm.cat([jaxm.stack([xp, yp]), statep_partial])
    #    return state + dt * statep

    # dynamics_path = default_dynamics_path if dynamics_path is None else Path(dynamics_path)
    # dynamics_data = json.loads(dynamics_path.read_text())
    # params = {k: jaxm.array(v) for (k, v) in dynamics_data["params"].items()}

    # @jaxm.jit
    # def dyn_fn(x, u, params):
    #    assert x.ndim == u.ndim
    #    assert x.shape[-1] == 11
    #    assert u.shape[-1] == 4
    #    org_shape = x.shape[:-1]
    #    x, u = x.reshape((-1, 11)), u.reshape((-1, 4))
    #    f = jaxm.vmap(_aero_dynamics, in_axes=(0, 0, None))(x, u, params)
    #    fx = jaxm.vmap(jaxm.jacobian(_aero_dynamics, argnums=0), in_axes=(0, 0, None))(x, u, params)
    #    fu = jaxm.vmap(jaxm.jacobian(_aero_dynamics, argnums=1), in_axes=(0, 0, None))(x, u, params)

    #    f = f.reshape(org_shape + f.shape[-1:])
    #    fx = fx.reshape(org_shape + fx.shape[-2:])
    #    fu = fu.reshape(org_shape + fu.shape[-2:])
    #    return f, fx, fu
    # return dyn_fn, params

    dynamics_path = Path(__file__).absolute().parents[1] / "data" / "dynamics_linear.json"
    dynamics_state = json.loads(dynamics_path.read_text())
    #fn = dynamics.int_f_fx_fu_fn
    fn = dynamics.f_fx_fu_fn
    params = {k: jaxm.array(v) for (k, v) in dynamics_state["params"].items()}
    #params["pos_ref"] = jaxm.zeros(3)
    #params["ang_ref"] = jaxm.zeros(3)
    f_fx_fu_fn = lambda x, u, *args: fn(x, u, params)

    return f_fx_fu_fn, params
