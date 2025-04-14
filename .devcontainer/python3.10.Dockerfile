ARG VARIANT="ubuntu22.04" 
ARG CUDA_VERSION="12.6.2"
ARG USERNAME="user"
ARG USER_UID="1000"
ARG USER_GID=${USER_UID}

# Start from a CUDA development image
FROM nvidia/cuda:${CUDA_VERSION}-devel-${VARIANT} AS build-env

ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG VARIANT
ARG CUDA_VERSION

ARG PROJDIR="/workspace/DRLearner_Beta"
# Make non-interactive environment.
ENV DEBIAN_FRONTEND=noninteractive

## Installing dependencies.
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    build-essential \
       python3.10 \
       python3.10-dev \
       python3-pip \
       python3-venv \
       libpython3.10 \
       wget \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       git \
       unar \
       zlib1g-dev \
       tmux \
       unrar \
       # docker: \
       ca-certificates \
       curl \
        # non-root user: \
        sudo \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    # docker: \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    # google cloud sdk: \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

WORKDIR /tmp
ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m pip install virtualenv \
    && python3 -m virtualenv --copies ${VIRTUAL_ENV} \
    && chown -R $USERNAME:$USERNAME ${VIRTUAL_ENV} /tmp \
    && sudo chmod +x ${VIRTUAL_ENV}/bin/activate
    
USER $USERNAME
SHELL ["/bin/bash", "-c"]

COPY requirements.txt  requirements.txt
RUN source ${VIRTUAL_ENV}/bin/activate \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && unlink requirements.txt \
    && python3 -m pip install git+https://github.com/ivannz/gymDiscoMaze.git@stable \
    && wget http://www.atarimania.com/roms/Roms.rar \
    && unrar e -y Roms.rar /tmp/roms/ \
    && unlink Roms.rar

COPY ./external/xm_docker.py ${VIRTUAL_ENV}/lib/python3.10/site-packages/launchpad/nodes/python/xm_docker.py
COPY ./external/launch.py ${VIRTUAL_ENV}/lib/python3.10/site-packages/launchpad/launch/xm_docker/launch.py
COPY ./external/vertex.py ${VIRTUAL_ENV}/lib/python3.10/site-packages/xmanager/cloud/vertex.py

FROM nvidia/cuda:${CUDA_VERSION}-devel-${VARIANT}

ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG PROJDIR

ENV DEBIAN_FRONTEND=noninteractive


# Copy apt sources and keyrings from the build stage
COPY --from=build-env /etc/apt/sources.list.d/ /etc/apt/sources.list.d/
COPY --from=build-env /usr/share/keyrings /usr/share/keyrings
COPY --from=build-env /etc/apt/trusted.gpg.d /etc/apt/trusted.gpg.d
COPY --from=build-env /etc/apt/keyrings/ /etc/apt/keyrings/
COPY --from=build-env /tmp/roms/ /tmp/roms/

RUN apt-get update -y \
    # python:
    && apt-get install -y python3.10-dev \
    # 
    xvfb ffmpeg \
    # docker:
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    git \
    # google cloud sdk:
    google-cloud-cli \
    # gpg and gpg-agent:
    gnupg2 gnupg-agent \
    # non-root user: 
    sudo \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    # && groupadd docker \
    && usermod -aG docker ${USERNAME} \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt \
    apt-get clean

USER ${USERNAME}
WORKDIR ${PROJDIR}

ENV VIRTUAL_ENV=/opt/venv

# Copy the virtual environment from the build stage
COPY --from=build-env ${VIRTUAL_ENV} ${VIRTUAL_ENV}

ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
ENV PYTHONPATH=${PROJDIR}:${PYTHONPATH:-}
