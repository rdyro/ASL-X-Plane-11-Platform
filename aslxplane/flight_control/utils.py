import sys
import struct
import time
from pathlib import Path
import random
import math
import numpy as np

from ..utils.robust_xpc import RobustXPlaneConnect
from ..perception.state import FlightState, deg2rad, rad2deg
from ..perception.vis_state import FlightStateWithVision
from ..perception.state import SIM_SPEED, SIM_TIME, BRAKE, ROTATION_SPEEDS, SPEEDS
from ..perception.state import LATLON_DEG_TO_METERS

####################################################################################################


def reset_flight(xp: RobustXPlaneConnect, on_crash_only:bool =True):
    """Send a command COMM to the XPlaneConnect plugin in the format of (COMM, len(cmd), cmd)."""
    is_crashed = xp.getDREF("sim/flightmodel2/misc/has_crashed")[0] > 0.0
    if is_crashed or not on_crash_only:
        for cmd in ["sim/operation/reset_flight"]:
            buffer = struct.pack(f"<4sxB{len(cmd)}s".encode(), b"COMM", len(cmd), cmd.encode())
            xp.sendUDP(buffer)
    time.sleep(4.0)


def sample_point_in_triangle(A, B, C):
    A, B, C = np.array(A), np.array(B), np.array(C)
    # Generate two random numbers between 0 and 1
    r1 = random.random()
    r2 = random.random()

    # Apply the formula for generating a random point in a triangle
    sqrt_r1 = math.sqrt(r1)
    point = (1 - sqrt_r1) * A + sqrt_r1 * (1 - r2) * B + sqrt_r1 * r2 * C

    return point
