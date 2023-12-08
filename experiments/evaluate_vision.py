from __future__ import annotations

import sys
from pathlib import Path
import random
from argparse import ArgumentParser
import json

import numpy as np
from tqdm import tqdm
from ray import tune
from ray.air import RunConfig

from aslxplane import xpc
from aslxplane.flight_control import utils
from aslxplane.flight_control.utils import RobustXPlaneConnect, FlightState, FlightStateWithVision
from aslxplane.flight_control.utils import deg2rad, rad2deg, reset_flight
from aslxplane.utils.video_recording import VideoRecorder
from aslxplane.flight_control.lqr_controller import LQRFlightController, DEFAULT_COST_CONFIG
from aslxplane.simulation.weather import set_the_weather


def landing_evaluation_cost_fn(x_hist, target, approach_ang):
    x_hist, target = np.array(x_hist), np.array(target)[:2]
    dx = target[:2] - x_hist[:, :2]
    min_approach_idx = np.argmin(np.linalg.norm(dx, axis=-1))
    approach_alt = x_hist[min_approach_idx, 2]
    if approach_alt > 25.0:
        return 1e9
    v_norm = np.array([np.cos(approach_ang), np.sin(approach_ang)])
    v_par = np.sum(dx * v_norm[None, :], -1)[:, None] * v_norm[None, :]
    v_perp = dx - v_par
    return float(np.linalg.norm(v_perp, axis=-1)[min_approach_idx])


MODEL_PATH = Path(__file__).absolute().parents[1] / "aslxplane" / "data" / "resnet50_model.pt"


def run_trial(
    cost_config,
    sim_speed=3.0,
    record_video=False,
    view=None,
    abort_at=1e9,
    offsets_to_test=[(0, 0)],
    display: bool = False,
    data_prefix: str | Path = ".",
    randomize_weather: bool = False,
    use_vision: bool = False,
):
    trial_id = random.randint(0, int(1e6) - 1)
    reset_flight(RobustXPlaneConnect(), on_crash_only=False)
    controller = LQRFlightController(
        config={
            "sim_speed": sim_speed,
            "use_vision": use_vision,
            "vision_config": {"camera_id": 2, "model_path": MODEL_PATH},
        },
        view=view,
    )
    hist_list, cost_list = [], []
    recorder = None
    for offset in tqdm(offsets_to_test):
        if randomize_weather:
            weather_desc = set_the_weather()
            weather_path = Path(data_prefix) / Path(f"recording_weather_{trial_id:06d}.json")
            weather_path.write_text(
                json.dumps({k: np.array(v).tolist() for (k, v) in weather_desc.items()})
            )
        controller.cost_config.update(cost_config)
        controller.config.update({"x0_offset": offset[0], "y0_offset": offset[1]})
        if record_video:
            recording_path = Path(data_prefix) / f"recording_{trial_id:06d}"
            recorder = VideoRecorder(recording_path, controller.flight_state, 2, display=display)
        crashed = controller.loop(how_long=abort_at)
        hist = {
            "x": [x.tolist() for x in controller.x_hist],
            "x_vis": [x.tolist() for x in controller.x_vis_hist],
            "u": [z.tolist() for z in controller.u_hist],
            "t": controller.t_hist,
        }
        Path("state_history.json").write_text(json.dumps(hist))
        controller.controller = "pid"

        controller.done = False
        if crashed:
            if recorder is not None:
                recorder.close()
            controller.close()
            return dict(objective=1e9)
        cost = landing_evaluation_cost_fn(
            hist["x"], target=controller.target, approach_ang=controller.approach_ang
        )
        cost_list.append(cost)
        print(f"cost = {cost:.4e}")
        hist_list.append(dict(offset=np.array(offset).tolist(), hist=hist))
        if cost > 1e8:
            break
    objective = np.max(cost_list)
    if recorder is not None:
        recorder.close()
    controller.close()
    return dict(
        objective=objective, cost_list=json.dumps(cost_list), hist_list=json.dumps(hist_list)
    )


####################################################################################################


def main():
    offsets_to_test = [
        utils.sample_point_in_triangle(*[(0.0, 0.0), (-2e3, 0.0), (2e3, 0.0)]) for _ in range(1)
    ]
    cost_config = {}
    offset = offsets_to_test[0]

    run_trial(
        cost_config,
        sim_speed=1.0,
        abort_at=120.0,
        record_video=True,
        display=False,
        view=xpc.ViewType.FullscreenNoHud,
        offsets_to_test=[offset],
        use_vision=True,
    )


if __name__ == "__main__":
    main()
