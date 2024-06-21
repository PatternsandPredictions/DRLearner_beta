# Docker image with tested GPU support.

Present docker image contains dockerfile, updated docker-compose and post-installation script for docker-based setup.

## Prerequisites
*****This setup was tested on Linix host machine only.** - most of the cloud setups will have variations of linux host.

In order to operate successfully with docker image, user should have local NVIDIA drivers installed as well as NVIDIA CUDA toolkit.
Both installations from ubuntu repo and according to NVIDIA instructions.

### Useful references:
1. [Download Nvidia drivers](https://www.nvidia.com/download/index.aspx)
2. [Installation guide for CUDA on linux host](https://www.cherryservers.com/blog/install-cuda-ubuntu)
3. [Official CUDA toolkit installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) - tested both installations on Ubuntu, AWS Linux AMI and installation with conda.

### Usage
1. Once Nvidia drivers, CUDA toolkit and nvcc are installed, installation can be validatied running the next commands.
```nvcc --version```
```nvidia-smi```

2. Copy Dockerfile, compose.yaml and post-install from current directory to root project directory.
3. Run ```docker-compose build --no-cache && docker-compose up -d```
4. Once system is fully built, enter the container, and check state of installed nvidia libs, using commands above.
5. Change permissions to post-install.sh files to be executable if needed.
6. Inside of container, run `./post-install.sh` to install the post-installation requirements.
7. Run any of examples.

