from __future__ import annotations

import sys
import time
import fcntl
import os
import subprocess
from subprocess import Popen, check_call, PIPE
from multiprocessing import Process, Event
import signal
from signal import SIGTERM
import logging

logging.basicConfig(level=logging.INFO)


def exit_now():
    [os.kill(int(p), SIGTERM) for p in os.popen(f"pgrep -P {os.getpid()}").read().split()]
    sys.exit(0)

def set_non_blocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def run_persistent(name: str, cmd: list[str], exit_event: Event) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logger = logging.getLogger(name)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    set_non_blocking(p.stdout)
    set_non_blocking(p.stderr)
    while not exit_event.is_set():
        # restart if necessary
        if p.poll() is not None:
            p = Popen(cmd, stdout=PIPE, stderr=PIPE)
            set_non_blocking(p.stdout)
            set_non_blocking(p.stderr)
        p.stdout.flush(), p.stderr.flush()
        try:
            lines = p.stdout.readlines()
            for line in lines:
                logger.info(line.decode("utf-8").rstrip())
        except IOError:
            pass
        try:
            lines = p.stderr.readlines()
            for line in lines:
                logger.warn(line.decode("utf-8").rstrip())
        except IOError:
            pass
        time.sleep(3e-1)
    p.kill()


####################################################################################################

OBS_WARNING_MESSAGE = """OBS Studio is not installed in your system or is not
available in the shell. Please, install it via `flatpak`

$ sudo apt install flatpak
$ flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
$ flatpak install flathub com.obsproject.Studio

or via apt (if on Debian/Ubutnu)

$ sudo apt install obs-studio
"""

def _which_obs_root():
    # check if obs is available in the path (apt installed)
    obs_cmd = None
    try:
        check_call(["which", "obs"], stderr=PIPE, stdout=PIPE)
        obs_cmd = ["obs"]
    except:
        pass
    if obs_cmd is not None:
        return obs_cmd

    # check if the flatpak version of obs is available
    try:
        check_call(["flatpak", "info", "com.obsproject.Studio"], stderr=PIPE, stdout=PIPE)
        obs_cmd = ["flatpak", "run", "com.obsproject.Studio"]
    except:
        pass
    return obs_cmd


def which_obs():
    obs_root = _which_obs_root()
    if obs_root is None:
        raise RuntimeError(OBS_WARNING_MESSAGE)
    return obs_root + [
        "--scene",
        "aslxplane", # we are selecting our manually created preset
        "--startvirtualcam", # virtual OpenCV-readable camera (v4l2 driver)
        "--minimize-to-tray", # start minimized
        "--disable-shutdown-check", # just for housekeeping
        "--multi", # to not prompt e.g. dead process detection
    ]


####################################################################################################

XPLANE_WARNING_MESSAGE = """You did not specify where the X-Plane executable is.
For instance if you have X-Plane installed in "~/X-Plane-11/X-Plane-x86_64", please do:

$ env XPLANE_EXEC_PATH="$HOME/X-Plane-11/X-Plane-x86_64" python3 -m aslxplane.launch
$ # or set 'export XPLANE_EXEC_PATH="$HOME/X-Plane-11/X-Plane-x86_64' 
$ # in your ~/.bashrc or ~/.zshrc, then
$ python3 -m aslxplane.launch
"""


def which_xplane():
    xplane_path = os.environ.get("XPLANE_EXEC_PATH", "X-Plane-x86_64")
    return [xplane_path]


####################################################################################################


def main():
    exit_event = Event()
    obs_cmd = which_obs()
    xplane_cmd = which_xplane()
    ps = [Process(target=run_persistent, args=("xplane", xplane_cmd, exit_event))]
    ps[0].start()
    time.sleep(1)
    if not ps[0].is_alive():
        print()
        print(XPLANE_WARNING_MESSAGE)
        print()
        exit_now()

    ps.append(Process(target=run_persistent, args=("obs", obs_cmd, exit_event)))
    ps[-1].start()

    try:
        while True:
            time.sleep(1e-1)
    except KeyboardInterrupt:
        pass
    print("Interrupt detected")
    exit_event.set()
    [p.join() for p in ps]
    exit_now()


if __name__ == "__main__":
    main()
