#!/usr/bin/env python3

from pathlib import Path
from shutil import copytree
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--xplane-dir", type=Path, required=True)

    args = parser.parse_args()
    msg = (
        "The path to the X-Plane you provided does not contain the folder `Resources`. "
        + "Double check that the path you provided is the path of the X-Plane "
        + "installation directory."
    )
    assert (Path(args.xplane_dir) / "Resources").exists(), msg
    plugin_dir = Path(args.xplane_dir) / "Resources" / "plugins"
    plugin_dir.mkdir(exist_ok=True)
    bundled_plugin = Path(__file__).parent / "aslxplane" / "data" / "XPlaneConnect"
    copytree(bundled_plugin, plugin_dir)


if __name__ == "__main__":
    main()
