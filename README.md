# ASL X-Plane 11 Test Platform

Based on the popular X-Plane 11 a photo-realistic aircraft simulator with accurate physics. 

We present a Python-based simulation platform to test and benchmark the
performance of perception and control algorithms on challenging control,
learning-based control and out-of-distribution control real-world problems in
flight.

This code build on the plugin 
[NASA-ULI-Xplane repo](https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator/tree/main). X-Plane 11 is . 

In our own work, we use it to test on 
  - vision-based autonomous taxiing
  - autonomous landing

<p align="center">
<img src="media/main_video.gif" style="width: 100%;max-width:900px"/>
</p>

# Quickstart

Running experiments often requires writing bespoke code. We do not attempt to
provide a one-size-fits-all solution. Instead, we suggest you take a look at:
- [experiments/landing_with_ground_truth.py](experiments/landing_with_ground_truth.py)
- [experiments/evaluate_vision.py](experiments/evaluate_vision.py) 
  - you need to download the model checkpoint via `$ git lfs pull`

We think of the `experiments` folder as the `examples` folder in other projects.

Our main landing controller is an LQR controller using learned, partially-linear dynamics (for Cessna Skyhawk).

We encourage you to copy and modify [aslxplane/flight_control/lqr_controller.py](aslxplane/flight_control/lqr_controller.py) to make your own controller!

---

# Installation

The intallation requires:
1. X-Plane 11
2. OBS installation and setup
3. install the XPlaneConnect plugin (via [install_plugin.py](install_plugin.py))
4. this package

### 1. Getting X-Plane 11

Most likely the best place to get X-Plane 11 is from Steam here: [https://store.steampowered.com/app/269950/XPlane_11/](https://store.steampowered.com/app/269950/XPlane_11/)

### 2. Installing OBS

We use OBS to capture the simulator window and expose it as a virtual camera device that can be accessed using OpenCV in Python.

We ask that you:
- install OBS
  - either via `apt` (on a debian/ubuntu) system: `$ sudo apt install obs-studio`
  - or via flatpak: `$ flatpak install flathub com.obsproject.Studio`
- install the virtual camera module `v4l2loopback-dkms`
  - on a debian/ubuntu system: `$ sudo apt install v4l2loopback-dkms`

### 2. Configure the recording settings in OBS

1. open OBS
2. create a new scene called `aslxplane`
3. add a new source called `X-Plane 11` of type `Window Capture`
    - select the window of the simulator
    - uncheck `Capture Cursor`
4. close OBS

### 3. Installing this package

You need to install the XPlaneConnect plugin. We made a script to do so:
```bash
$ python3 install_plugin.py --xplane-dir {/path/to/X-Plane-11-directory}
```

### 4. Installing this package

Install with `pip`
```bash
$ pip install -e . # `-e` for editable
$ # it'll let you edit the code in place for development
```

---

# Launching X-Plane 11 and OBS with ease

We provide a launch script that launches X-Plane 11 and OBS and starts the virtual camera. Should either of the programs crash, the script will restart them.

You can do so by running:
```bash
$ export XPLANE_EXEC_PATH={/path/to/X-Plane-x86_64}
$ python3 -m aslxplane # after installing this package
```

# State and Dynamics

We define the state of the aircraft using the 11-dimensional state vector (skipping side-slip velocity).
- `x`: position in the North direction (meters)
- `y`: position in the East direction (meters)
- `z`: position in the Down direction (meters)
- `v`: horizontal airspeed (meters/second)
- `vh`: vertical speed (meters/second)
- `pitch`: pitch angle (radians)
- `roll`: roll angle (radians)
- `yaw`: yaw angle (radians)
- `dpitch`: pitch rate (radians/second)
- `droll`: roll rate (radians/second)
- `dyaw`: yaw rate (radians/second)

X-Plane 11 does not expose the underlying dynamics of the aircraft. Instead, we
use a learned partially linearized model of the aircraft in
[aslxplane/flight_control/dynamics.py](aslxplane/flight_control/dynamics.py).


# Efficient Vision 

<p align="center">
<img src="media/vision_diagram_obs_opencv.png" style="width: 100%;max-width:500px"/>
</p>

In order to offer efficient and hassle free vision-based perception, we need to
capture the output of the simulator and expose it to the user.

We do this by capturing the window of the simulator using [OBS Studio](https://obsproject.com/) and creating a virtual camera device that can be efficiently accessed using OpenCV in Python.

<p align="center">
<img src="media/vision_estimation.gif" style="width: 100%;max-width:900px"/>
</p>

We explore training a model to estimate the position of the aircraft without GPS information from vision in [experiments/evaluate_vision.py](experiments/evaluate_vision.py).


# Controlling the Weather

We offer a simplified interface for setting or randomizing the weather. Take a look at
[aslxplane/simulation/weather.py](aslxplane/simulation/weather.py) for more details.

<p align="center">
<img src="media/weather_landing.gif" style="width: 100%;max-width:900px"/>
</p>

---

<!-- html vertical space -->
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p align="center">Old README</p>

---


# ASL X-Plane 11 Simulator
Based on the popular X-Plane 11 aircraft simulator, we present a convenient Python-based simulation platform to test and benchmark the performance of perception and control algorithms when they experience Out-of-Distribution scenarios in closed-loop. This platform is refactored from the original [NASA-ULI-Xplane repo](https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator/tree/main) to expand functionality, improve code quality, readability, and modularity. We offer photo-realistic graphics and accurate physics simulation Currently, the simulator offers a single control task: Vision-based autonomous taxiing. 

Users can flexibly define different OOD scenarios based on weather that may cause vision-degradation. Currently, we support variations in/addition of:
- cloud levels		
- time-of-day		
- additive image noise		
- snow		
- snowfall		
- rain 		
- motion-blur
- combinations of the above		
- several severity levels per corruption type
- linearly increasing and decreasing severity throughout an episode

### Features offered:
1. Conveniently interact with the simulator through the `XPlaneBridge` Python API, similar to the CARLA client.
2. Modular and standardized abstractions for perception and estimation to facilitate development of control/perception algorithms or use of existing systems
3. Specify and run thousands of simulations by modifying example `yaml` param files to sample environment variations
4. A single lightweight example script for sampling environments and running and recording experiments
5. Some utilities to analyze data and create videos of all episodes in an experiment
6. Track episode statistics for series of experiments anywhere through the web using [weights and biases](https://wandb.ai/)

## Setup Instructions:
1. First, follow the legacy setup instructions, available at `https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator`. Specifically, follow the legacy instructions to:
    - Download and install the X-Plane 11 simulator
    - Install the X-Camera Plugin
    - Download the 208B Grand Caravan Aircraft model
    - Download and install the X-Plane connect plugin
    - Configure the flight details in X-Plane
    - Configure the X-Camera in X-Plane
2. Clone the repository: `git clone https://github.com/StanfordASL/XPlane-ASL.git`
3. Enter the XPLANE-ASL directory: `cd XPLANE-ASL`
4. (optional) create and activate a virtual environment to install the packages:
    - to create virtual env: `python3 -m venv xplane-env --system-site-packages`
    - to activate virtual env: `source xplane-env/bin/activate`
    - to shut down virtual env: `deactivate`
    - make sure you add the name of your virtual environment to the .gitignore file! (xplane-env is already included)
5. build the aslxplane and xpc3 packages
    - ensure build package is up-to-date: `python3 -m pip install --upgrade build`
    - build packages: `python3 -m build`
6. install the image augmentation library:
    - `pip3 install imagecorruptions`
    - `pip3 install git+https://github.com/marcown/imgaug.git` (May 2023: do not use the pypi package index version, it has a bug introduced by the latest numpy version)
6. install the aslxplane and xpc3 packages: `python3 -m pip install -e ./`
7. If you create new files or packages, uninstall, rebuild, and reinstall all the packages in the reopository:
    - uninstall packages `python3 -m pip uninstall ./`
    - rebuild packages (step 4)
    - reinstall packages (step 5)

## Quick Start Workflow:
Quick-start workflow to run an experiment:
1. create a folder to store your experiment data: `mkdir your-data-directory/your-experiment-name`
2. copy the template parameter files into your experiment directory:
    - `mkdir your-data-directory/your-experiment-name/params `
    - `cp Xplane-ASL/params/simulator_params.yaml your-data-directory/your-experiment-name/params`
    - `cp Xplane-ASL/params/experiment_params.yaml your-data-directory/your-experiment-name/params`
    Note: if running data-collection to train perception models, use the `sinusoid_dataset_params.yaml` instead of `experiment_params.yaml` template
3. modify the parameter files to your desired settings
4. (optionally) to dry-run your experiment setup, initially consider running with params `debug/perception` set to `True` and `logging/log_data` set to `False` and look at the experiment runs to see if it matches desired behavior
5. enter the XPLANE-ASL directory: `cd XPlane-ASL/`
6. run your experiment by calling `trajectoryrunner.py` with your experiment directory as an argument:
    - `python3 trajectoryrunner.py relative-path-to-data-dir/your-data-directory/your-experiment-name/`

Download datasets and quickstart model checkpoints [here](https://drive.google.com/drive/folders/1n_RRKtvQqthSMuK0vbgHUMBw_FAvuwWZ?usp=sharing).

## Citation
If you use our simulator in your work, please cite its usage as:
```latex 
@article{SinhaSchmerlingPavone2023,
    title={Closing the Loop on Runtime Monitors with Fallback Safe MPC},
    author={Sinha, Rohan and Schmerling, Ed and Pavone, Marco},
    journal={arXiv preprint arXiv:tbd},
    year={2023}
}
```

## To Do:
- [ ] generate mkdocs
- [ ] figure out how to place obstacles on the runway
- [ ] change the toolbar crop pixels from a fixed value to a fraction of the monitor size such that trained xplane controllers can work for many different monitor setups...

Nice-to-haves:
- [ ] implement wrapper to conform to openai gym api specification
- [ ] implement functionality to run the xplane_bridge as a ROS node
