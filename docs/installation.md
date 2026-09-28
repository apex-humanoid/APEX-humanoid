# Installation

Requires Linux, an NVIDIA GPU, Git, Isaac Sim 4.2.0 and Python 3.10. Isaac Lab 1.3.0 and RSL-RL are included.

## Isaac Sim binary installation

Install Isaac Sim 4.2.0, then run these commands without activating another Conda environment:

```bash
git clone https://github.com/apex-humanoid/APEX-humanoid.git
cd APEX-humanoid
ln -s /path/to/isaac-sim _isaac_sim
./isaaclab.sh -i rsl_rl
```

Replace `/path/to/isaac-sim` with the directory containing `python.sh`. The launcher uses the simulator's bundled Python.

## Pip installation

The pip packages require glibc 2.34 or newer (`ldd --version`). On minimal Ubuntu containers, install system dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y git libxt6
```

Create a Python 3.10 environment and install the packages:

```bash
git clone https://github.com/apex-humanoid/APEX-humanoid.git
cd APEX-humanoid
conda create -n apex-humanoid python=3.10 -y
conda activate apex-humanoid
python -m pip install --upgrade pip
python -m pip install \
  'torch==2.4.0+cu118' 'torchvision==0.19.0+cu118' \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install \
  'isaacsim-rl==4.2.0.2' 'isaacsim-replicator==4.2.0.2' \
  'isaacsim-app==4.2.0.2' 'isaacsim-extscache-physics==4.2.0.2' \
  'isaacsim-extscache-kit-sdk==4.2.0.2' 'isaacsim-extscache-kit==4.2.0.2' \
  -c constraints.txt --extra-index-url https://pypi.nvidia.com
python -c 'import isaacsim'
./isaaclab.sh -i rsl_rl
```

Accept NVIDIA's EULA when prompted. Initial startup may download extensions.

## Containers

Enable NVIDIA GPU and graphics driver access. For a custom pip container, include the Vulkan and EGL driver registrations when the source files exist on the host:

```bash
--gpus all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
--mount type=bind,src=/usr/share/vulkan/icd.d/nvidia_icd.json,dst=/etc/vulkan/icd.d/nvidia_icd.json,readonly \
--mount type=bind,src=/usr/share/glvnd/egl_vendor.d/10_nvidia.json,dst=/usr/share/glvnd/egl_vendor.d/10_nvidia.json,readonly
```

These are options for `docker run`, placed before the image name. To select one GPU, use `--gpus device=0` with the desired device index.

See the [README](../README.md) for training and [pretrained policy playback](checkpoints.md) for inference.
