FROM nvidia/cuda:11.1.1-base-ubuntu20.04
# FROM python:3.7
## Basic dependencies.
ADD . /app

WORKDIR /app

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

### Installing dependencies.
RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
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
       libpython3.7

# Conda environment
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# SHELL ["/bin/bash", "-c"]
RUN conda create --name drlearner python=3.7 -y
RUN python --version
RUN echo "source activate drlearner" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib:/usr/lib:/usr/local/lib:~/anaconda3/envs/drlearner/lib:/opt/conda/envs/drlearner/lib:/opt/conda/lib

ENV PYTHONPATH=$PYTHONPATH:$(pwd)
RUN conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib:/usr/lib:/usr/local/lib:~/anaconda3/envs/drlearner/lib:/opt/conda/envs/drlearner/lib:/opt/conda/lib
RUN conda env config vars set PYTHONPATH=$PYTHONPATH:$(pwd)
RUN conda install nvidia/label/cuda-11.3.1::cuda-nvcc -y
RUN conda install -c conda-forge cudatoolkit=11.3.1 cudnn=8.2 -y
RUN conda install -c anaconda git

RUN chmod +x ./

CMD ["/bin/bash"]


