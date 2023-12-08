from __future__ import annotations

import os
import time
from pathlib import Path
import json
import math
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Optional
from multiprocessing import Lock

os.environ["JAX_PLATFORM_NAME"] = "CPU"

import numpy as np

from .. import xpc
from . import dynamics, utils
from .lqr_utils import design_LQR_controller
from .utils import RobustXPlaneConnect, deg2rad, rad2deg, FlightState, reset_flight
from .utils import FlightStateWithVision
from .utils import LATLON_DEG_TO_METERS

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "CPU"
os.environ["JAX_ENABLE_X64"] = "True"

from jaxfi import jaxm

jaxm.set_default_dtype(np.float64)

####################################################################################################

DEFAULT_CONFIG = {
    "sim_speed": 1.0,
    "x0_offset": 0.0,
    "y0_offset": 0.0,
}

# magic numbers for Cessna Skyhawk
DEFAULT_COST_CONFIG = {
    "heading_cost": 1e4,
    "roll_cost": 3e4,
    "position_cost": 1e0,
    "altitude_cost": 1e2,
    "par_cost": 498.863996,
    "perp_cost": 481.605499,
    "perp_quad_cost": 0.002698,
    "par_quad_cost": 1e-3,
}

# weird angle misalignment vs runway, perhaps due to in-game magnetic compass distortion
ANGLE_CORRECTION = -0.1


class LQRFlightController:
    def __init__(
        self,
        config: dict[str, Any] = DEFAULT_CONFIG,
        cost_config: dict[str, float] = DEFAULT_COST_CONFIG,
        view: Optional[xpc.ViewType] = None,
        angle_correction: float = ANGLE_CORRECTION,
    ):
        """LQR Flight Controller - mostly tested for landing or heading line navigation.

        Args:
            config (dict[str, Any], optional): Defaults to DEFAULT_CONFIG.
            cost_config (dict[str, float], optional): LQR cost config. Defaults
                                                      to DEFAULT_COST_CONFIG.
            view (Optional[xpc.ViewType], optional): Which xpc.ViewType to use
                                                     on the plane. Defaults to None.
            angle_correction (float, optional): Landing strip angle relative to
                                                flight reset position. Defaults
                                                to ANGLE_CORRECTION.
        """

        self.config, self.cost_config = deepcopy(DEFAULT_CONFIG), deepcopy(DEFAULT_COST_CONFIG)
        self.config.update(deepcopy(config))
        self.cost_config.update(deepcopy(cost_config))

        self.xp = RobustXPlaneConnect()
        self.int_state = np.zeros(6)
        self.vis_flight_state, self.use_vision = None, self.config.get("use_vision", None)
        self.flight_state = FlightState()
        if self.use_vision:
            self.vis_flight_state = FlightStateWithVision(
                **self.config.get("vision_config", dict())
            )
        while not np.all(np.isfinite(self.flight_state.last_sim_time_and_state[1])):
            time.sleep(1e-1)
        if self.use_vision:
            # wait for the vision state to initialize e.g., a deep model
            while not np.all(np.isfinite(self.vis_flight_state.last_sim_time_and_state[1])):
                time.sleep(1e-1)

        # initialize reference control as a return to initial state
        time.sleep(0.3)
        self.state0, self.posi0 = self.get_curr_state(), self.xp.getPOSI()
        self.target = self.get_curr_state()
        self.view = view if view is not None else xpc.ViewType.Chase

        # touchdown control parameters
        self.v_ref = 50.0
        self.params = dict()
        self.params["pos_ref"] = np.array([0.0, 0.0, 300.0])
        self.params["ang_ref"] = np.array([deg2rad(0.0), 0.0, deg2rad(self.posi0[5])])

        self.set_brake(0)
        self.t_start = time.time()
        self.u_hist, self.x_hist, self.x_vis_hist, self.t_hist = [], [], [], []
        self.controller = "lqr"
        self._read_dynamics()

        # runway specific angle correction
        self.approach_ang = deg2rad(self.posi0[5]) + angle_correction
        self.lock = Lock()
        self.ts, self.X, self.U, self.Ls = None, None, None, None
        self.done = False
        self.reset()

        self.data = dict()

    def loop(self, how_long: float = math.inf) -> None:
        """Apply control in a loop.

        Args:
            how_long (float, optional): How "real-time" long to run the loop at.
                                        Defaults to math.inf.
        """

        t_loop_start = time.time()
        self.reset()

        self.data = dict()
        self.it = 0
        self.u_hist, self.x_hist, self.t_hist = [], [], []
        self.t_start = time.time()

        # T = 100.0
        # T = 110.0 / self.config["sim_speed"]
        dt_small = 1.0 / 50.0
        t_prev = 0.0
        while not self.done and time.time() - t_loop_start < how_long:
            t_prev = time.time()
            self.apply_control()
            sleep_for = max(0, dt_small - (time.time() - t_prev))
            time.sleep(sleep_for)
            self.it += 1
            is_crashed = self.xp.getDREF("sim/flightmodel2/misc/has_crashed")[0] > 0.0
            if is_crashed:
                reset_flight(self.xp)
                return True
        if self.done:
            self.reset()
        return False

    @staticmethod
    def _build_control(pitch=0, roll=0, yaw=0, throttle=0, gear=0, flaps=0):
        return [min(max(x, -1), 1) for x in [pitch, roll, yaw]] + [
            min(max(x, 0), 1) for x in [throttle, gear, flaps]
        ]

    def set_brake(self, brake: float = 1) -> None:
        self.xp.sendDREF(utils.BRAKE, brake)

    def reset(self):
        """Reset the simulation to the state about 5km in the air behind the runway."""
        self.xp.sendDREF(utils.SIM_SPEED, self.config["sim_speed"])
        self.xp.sendVIEW(self.view)
        for _ in range(1):
            # arrest speed
            self.xp.sendPOSI(self.posi0)
            self.xp.sendDREFs(list(utils.SPEEDS.values()), [0 for _ in utils.SPEEDS.values()])
            # arrest rotation
            self.xp.sendDREFs(
                list(utils.ROTATION_SPEEDS.values()), [0 for _ in utils.ROTATION_SPEEDS.values()]
            )
            self.xp.sendPOSI(self.posi0)
            self.xp.sendCTRL(self._build_control())
            self.set_brake()

            posi = list(copy(self.posi0))
            posi[2] = 300
            dist = 6e3
            # posi[0] += dist / LONLAT_DEG_TO_METERS * -math.cos(deg2rad(posi[5])) + 3e3 / DEG_TO_METERS
            # posi[1] += dist / LONLAT_DEG_TO_METERS * -math.sin(deg2rad(posi[5])) + 3e3 / DEG_TO_METERS
            posi[0] += (
                dist / LATLON_DEG_TO_METERS * -math.cos(deg2rad(posi[5]))
                + self.config["x0_offset"] / LATLON_DEG_TO_METERS
            )
            posi[1] += (
                dist / LATLON_DEG_TO_METERS * -math.sin(deg2rad(posi[5]))
                + self.config["y0_offset"] / LATLON_DEG_TO_METERS
            )

            # set the plane at the new reset position, match simulation speed to heading
            self.xp.sendPOSI(posi)
            v = 60.0
            vx, vz = v * math.sin(deg2rad(self.posi0[5])), v * -math.cos(deg2rad(self.posi0[5]))
            self.xp.sendDREFs([utils.SPEEDS["local_vx"], utils.SPEEDS["local_vz"]], [vx, vz])
            time.sleep(0.5)
        self.data = dict()

    def get_time_state(self):
        if self.use_vision:
            return tuple(self.vis_flight_state.last_sim_time_and_state)
        return tuple(self.flight_state.last_sim_time_and_state)

    def get_curr_time(self):
        return self.flight_state.last_sim_time_and_state[0]

    def get_curr_state(self, vision: bool = False) -> np.array:
        if vision and self.use_vision:
            state = self.vis_flight_state.last_sim_time_and_state[1]
        else:
            state = self.flight_state.last_sim_time_and_state[1]
        state = np.concatenate([np.array(state), self.int_state])
        return state

    ################################################################################

    def _read_dynamics(self):
        dynamics_path = Path(__file__).absolute().parents[1] / "data" / "dynamics_linear.json"
        dynamics_state = json.loads(dynamics_path.read_text())
        fn = dynamics.int_f_fx_fu_fn
        self.params.update({k: jaxm.array(v) for (k, v) in dynamics_state["params"].items()})
        params = copy(self.params)
        self.f_fx_fu_fn = lambda x, u, *args: fn(x, u, params)

    def advance_state(self, dt):
        state = self.get_curr_state()
        pos_int = self.int_state[:3] + dt * (np.array(state[:3]) - self.params["pos_ref"])
        ang_int = self.int_state[3:6] + dt * (np.array(state[5:8]) - self.params["ang_ref"])
        int_state = np.concatenate([np.array(pos_int), np.array(ang_int)])
        self.int_state = 0.99**dt * int_state

    ################################################################################

    def _construct_lqr_problem(self, x0):
        # read in the target #######################################################################
        x_ref = np.copy(x0)
        target = self.target[:2] + 400 * np.array(  # 400 meters down the runway from starting point
            [math.cos(self.approach_ang), math.sin(self.approach_ang)]
        )
        dist = np.linalg.norm(target[:2] - x0[:2])
        # read in the target ######################################################################

        # create the cost weighting for state ######################################################
        cc = self.cost_config
        q_diag = (
            np.array(
                [cc["position_cost"], cc["position_cost"], cc["altitude_cost"]]
                + [1e3, 1e0]
                + [1e0, cc["roll_cost"], cc["heading_cost"]]
                + [1e-3, 1e-3, 1e-3]
                + [0 * 1e-3, 0 * 1e-3, 0 * 1e-3]
                + [0 * 1e-3, 0 * 1e-3, 0 * 1e-3]
            )
            / 1e3
        )
        Q = np.diag(q_diag)
        # create the cost weighting for state ######################################################

        # create the state reference ###############################################################
        v_norm = np.array([math.cos(self.approach_ang), math.sin(self.approach_ang)])
        dx = np.array(target[:2]) - np.array(x0[:2])
        v_par = np.sum(dx * v_norm) * v_norm
        v_perp = dx - v_par
        d_par = math.sqrt(max(5e2**2 - np.linalg.norm(v_perp) ** 2, 0)) / np.linalg.norm(v_par)
        x_ref[:2] = (
            x0[:2]
            + max(np.linalg.norm(v_perp), 1e2) * v_perp / np.linalg.norm(v_perp)
            + d_par * v_par
        )
        x_ref[2] = min(
            max(self.posi0[2], self.params["pos_ref"][2] * (dist / 5e3)), 300.0
        )  # altitude
        x_ref[3:5] = self.v_ref, 0.0  # velocities
        x_ref[5:8] = self.params["ang_ref"]
        x_ref[8:11] = 0  # dangles
        x_ref[11:] = 0  # integrated errors
        # create the state reference ###############################################################

        # augment the cost using automatic differentiation of an Huber-loss-like objective function
        if "cost_approx" not in self.data:

            def cost_fn(x0, target, v_norm):
                """Compute a position cost as a scalar."""
                dx = target[:2] - x0[:2]
                v_par = jaxm.sum(dx * v_norm) * v_norm
                v_perp = dx - v_par
                v_perp_norm = jaxm.linalg.norm(v_perp)
                v_perp_norm2 = jaxm.sum(v_perp**2)
                v_par_norm = jaxm.linalg.norm(v_par)
                cc = self.cost_config
                Jv_perp = jaxm.where(
                    v_perp_norm > 1e3, v_perp_norm, cc["perp_quad_cost"] * v_perp_norm2
                )
                Jv_par = v_par_norm
                return cc["perp_cost"] * Jv_perp + cc["par_cost"] * Jv_par

            @jaxm.jit
            def cost_approx(x0, target, v_norm):
                """Develop a quadratic approximation of the cost function based on a scalar cost."""
                g = jaxm.grad(cost_fn, argnums=0)(x0, target, v_norm)
                H = jaxm.hessian(cost_fn, argnums=0)(x0, target, v_norm)
                Q = H + 1e-3 * jaxm.eye(H.shape[-1])
                ref = x0 - jaxm.linalg.solve(Q, g)
                return Q, ref

            self.data["cost_fn"] = cost_fn
            self.data["cost_approx"] = cost_approx
        Qx, refx = self.data["cost_approx"](x0[:2], np.array(target)[:2], np.array(v_norm))
        x_ref[:2] = refx[:2]
        Q[:2, :2] = Qx[:2, :2] / 1e3
        # augment the cost using automatic differentiation of an Huber-loss-like objective function

        # create the control weight and reference ##################################################
        R = np.diag(np.array([1e0, 3e-1, 1e2, 1e0])) * 1e-1
        u_ref = np.array([0.0, 0.0, 0.0, 0.0])
        # create the control weight and reference ##################################################

        # normalize the cost for numerical stability #######################
        norm = np.linalg.norm(Q[:, :]) + np.linalg.norm(R[:, :])
        Q, R = Q / norm, R / norm
        # normalize the cost for numerical stability #######################

        return Q, R, x_ref, u_ref

    ################################################################################

    def apply_control(self):
        """Compute and apply the control action."""

        state = self.get_curr_state()
        vis_state = self.get_curr_state(vision=True)

        if self.controller == "pid":
            pitch, roll, heading = state[5:8]
            pitch_ref, roll_ref, heading_ref = deg2rad(5.0), 0.0, self.state0[7]
            u_pitch = -1.0 * (pitch - pitch_ref)
            u_roll = -1.0 * (roll - roll_ref)
            u_heading = -1.0 * (30.0 / state[3]) * (heading - heading_ref)
            throttle = 0.7
            u = np.array([u_pitch, u_roll, u_heading, throttle])
        elif self.controller == "lqr":
            Q, R, x_ref, u_ref = self._construct_lqr_problem(state)
            u0 = np.zeros(R.shape[-1])
            f, fx, fu = self.f_fx_fu_fn(state, u0)
            A, B, d = fx, fu, f - fx @ state - fu @ u0
            L, l = design_LQR_controller(A, B, d, Q, R, x_ref, u_ref, T=10)
            u = L @ state + l
        u_pitch, u_roll, u_heading, throttle = np.clip(u, [-1, -1, -1, 0], [1, 1, 1, 1])

        # landing stage, a poor man's finite state machine #########################################
        if state[2] < 5.0 or self.data.get("fixed_pitch", None) is not None:
            self.data.setdefault("fixed_pitch", u_pitch - 0.05)
            u_pitch, u_roll, u_heading, throttle = self.data["fixed_pitch"], 0.0, 0.0, 0.0
            self.set_brake(1)
        # landing stage, a poor man's finite state machine #########################################

        self.t_hist.append(self.get_curr_time())
        self.x_hist.append(copy(state))
        self.x_vis_hist.append(copy(vis_state))
        self.u_hist.append(copy(u))
        ctrl = self._build_control(pitch=u_pitch, roll=u_roll, yaw=u_heading, throttle=throttle)
        self.xp.sendCTRL(ctrl)

    def close(self):
        self.flight_state.close()
        self.done = True


####################################################################################################
