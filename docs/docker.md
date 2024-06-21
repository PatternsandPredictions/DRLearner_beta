# Running in Docker

Clone the repo
```
git clone https://github.com/PatternsandPredictions/DRLearner_beta.git
cd DRLearner_beta/
```

Install Docker (if not already installed) and Docker Compose (optional)
```
https://docs.docker.com/desktop/install/linux-install/
https://docs.docker.com/compose/install/linux/
```

1. Use Dockerfile directly 
```
docker build -t drlearner:latest .
docker run -it --name drlearner -d drlearner:latest
```
2. Use Docker compose 
```
docker compose up
```

Now you can attach yourself to the docker container to play with it.
```
docker exec -it drlearner bash
```
## Dockerfile
Using python image, setting "/app" as workdir and running essential linux dependecies
```
FROM python:3.10
ADD . /app
WORKDIR /app

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
```
Downloading conda and creating enviroment.
```
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN conda create --name drlearner python=3.10 -y
RUN python --version
RUN echo "source activate drlearner" > ~/.bashrc
ENV PATH /opt/conda/envs/drlearner/bin:$PATH
```

Installing requirements for python and downloading game roms.
```
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/ivannz/gymDiscoMaze.git@stable

# Get binaries for Atari games
RUN wget http://www.atarimania.com/roms/Roms.rar
RUN unar Roms.rar
RUN mv Roms roms
RUN ale-import-roms roms/
```

Setting up enviromental variables and changing acces mode for all files.
```
ENV PYTHONPATH=$PYTHONPATH:$(pwd)
RUN chmod -R 777 ./
```

Default command for running container. Here you can modify it or simply run with "CMD ["/bin/bash"]" and attach yourself to container to run commands directly. 
```
CMD ["python3" ,"examples/run_atari.py", "--level","PongNoFrameskip-v4", "--num_episodes", "1000", "--exp_path", "experiments/test_pong/", "--exp_name", "my_first_experiment"]
or 
CMD ["/bin/bash"]
```

## Docker compose
Compose run one service called drlearner that is built using Dockerfile present in the main directory. Thanks to setting volumes as .:/app we don't have to rebuilt container each time we change codebase. Setting flags stdin_open and tty allows interactive mode of docker container. Thanks to that option user can attach themselves to the container and use it interactivly.

```
services:
  drlearner:
    build: .
    volumes:
      - .:/app
    stdin_open: true # docker run -i
    tty: true        # docker run -t
    env_file:
      - .env
```
All the enviromental variable will be read from .env file.
