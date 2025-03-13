FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y wget git
RUN apt update && apt upgrade -y && apt install screen speedtest-cli git-lfs ffmpeg -y

# PM2
RUN apt-get update && apt-get install -y curl \
    && curl -sL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pm2

# Installing Miniconda    
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod 700 Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b \
    && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda init
# RUN conda create -n venv python=3.10.13 -y
# ENV PATH="/root/miniconda3/envs/venv_image/bin:${PATH}"
# RUN conda activate venv
    
RUN apt-get -y full-upgrade \
    && apt-get -y install python3-dev \
    && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip\
    apt-utils \
    curl \
    wget \
    vim \
    sudo \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-tk \
    python3-dev \
    git-lfs \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip \
    && pip install --upgrade pip setuptools \
    && pip install -r /workspace/requirements.txt  \    
    && rm -rf /root/.cache/pip/*

COPY . /workspace/

RUN chmod +x /workspace/*.sh

