FROM ubuntu:18.04 as base

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    python3-dev \
    libffi6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    nvidia-cuda-toolkit \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install build-essential -y \
    && apt-get install manpages-dev -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN apt-get update \
    && apt-get install python3.5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /3d_forest/miniconda3 \
    && rm Miniconda3-latest-Linux-x86_64.sh

# making conda env
ENV PATH=/3d_forest/miniconda3/bin:$PATH
RUN conda update -n base -c defaults conda \
    && conda create -n 3d_forest python=3.5 \
    && conda clean -ya


# Activate conda env
ENV PATH=/3d_forest/miniconda3/envs/3d_forest/bin:$PATH
RUN echo "source activate 3d_forest" > ~/.bashrc
ENV CONDA_DEFAULT_ENV=3d_forest



COPY ./requirements.txt /3d_forest/requirements.txt
COPY ./requirements_seg.txt /3d_forest/requirements_seg.txt  

RUN pip install --upgrade pip \
    && pip install -r /3d_forest/requirements.txt \
    && pip install -r /3d_forest/requirements_seg.txt \
    && pip install jupyter \
    && rm -rf /root/.cache/pip

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
