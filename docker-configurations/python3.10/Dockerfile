# Start from a CUDA development image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
 
# Make non-interactive environment.
ENV DEBIAN_FRONTEND noninteractive
 
## Installing dependencies.
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    build-essential \
       python3.10 \
       python3.10-dev \
       python3-pip \
       curl \
       wget \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       git \
       unar \
       libpython3.10 \
       zlib1g-dev \
       tmux \
    && rm -rf /var/lib/apt/lists/*
 
## Workdir
ADD . /app
WORKDIR /app
# Library paths.
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib:/usr/lib/:$(pwd)
ENV PYTHONPATH=$PYTHONPATH:$(pwd)
 
# Update pip to the latest version & install packages.
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install jax==0.4.3
RUN python3.10 -m pip install jaxlib==0.4.3+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
RUN python3.10 -m pip install git+https://github.com/ivannz/gymDiscoMaze.git@stable
 
# Atari games.
RUN wget http://www.atarimania.com/roms/Roms.rar
RUN unar Roms.rar
RUN mv Roms roms
RUN ale-import-roms roms/
 
RUN chmod +x ./
 
CMD ["/bin/bash"]
