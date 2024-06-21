FROM python:3.10
## Basic dependencies.
ADD . /app

WORKDIR /app

### Installing dependencies.
RUN apt-get update \
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
       unar \
       libpython3.10 \
       tmux 

# Conda environment

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# SHELL ["/bin/bash", "-c"]
RUN conda create --name drlearner python=3.10 -y
RUN python --version
RUN echo "source activate drlearner" > ~/.bashrc
ENV PATH /opt/conda/envs/drlearner/bin:$PATH

# RUN conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib:/usr/lib:/usr/local/lib:~/anaconda3/envs/drlearner/lib:/opt/conda/envs/drlearner/lib:/opt/conda/lib
# RUN conda env config vars set PYTHONPATH=$PYTHONPATH:$(pwd)

# Install dependencies (some of them are old + maybe there is need to check support of cuda)
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
RUN python3.10 -m pip install git+https://github.com/ivannz/gymDiscoMaze.git@stable
RUN conda install conda-forge::ffmpeg

# RUN pip install git+https://github.com/google-deepmind/acme.git@4c6351ef8ff3f4045a9a24bee6a994667d89c69c

    
# RUN conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0

# Get binaries for Atari games
RUN wget http://www.atarimania.com/roms/Roms.rar
RUN unar Roms.rar
RUN mv Roms roms
RUN ale-import-roms roms/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib:/usr/lib:/usr/local/lib:~/anaconda3/envs/drlearner/lib:/opt/conda/envs/drlearner/lib:/opt/conda/lib
ENV PYTHONPATH=$PYTHONPATH:$(pwd)
ENV XLA_PYTHON_CLIENT_PREALLOCATE='0'

# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib:/usr/lib:/usr/local/lib:~/anaconda3/envs/drlearner/lib
RUN chmod -R 777 ./


# CMD ["python3" ,"examples/run_lunar_lander.py"]
CMD ["/bin/bash"]

# CMD ["/bin/bash python3", "examples/run_atari.py --level PongNoFrameskip-v4 --num_episodes 1000 --exp_path experiments/test_pong/"]

# sudo docker build  -t rdlearner:latest .