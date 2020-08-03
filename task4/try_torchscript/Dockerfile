# https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/

# https://www.tensorflow.org/install/gpu

FROM nvidia/cudagl:10.1-devel-ubuntu18.04
# FROM nvcr.io/nvidia/tensorrt:20.06-py3
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y sudo locales

# Create user ubuntu and add it as a sudoer with no password
RUN adduser --disabled-password --shell /bin/bash --gecos "User" ubuntu && \
    sudo adduser ubuntu sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    sudo locale-gen en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=7.6.5.32-1+cuda10.1  \
    libcudnn7-dev=7.6.5.32-1+cuda10.1

# Install TensorRT. Requires that libcudnn7 is installed above.
RUN apt-get update && apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1

RUN apt-get update && apt-get install -y build-essential unzip pkg-config
RUN apt-get update && apt-get install -y libjpeg-dev libpng-dev libtiff-dev
RUN apt-get update && apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get update && apt-get install -y libv4l-dev libxvidcore-dev libx264-dev
RUN apt-get update && apt-get install -y libgtk-3-dev
RUN apt-get update && apt-get install -y libatlas-base-dev gfortran
RUN apt-get update && apt-get install -y python3-dev python3-pip
RUN apt-get update && apt-get install -y libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libdc1394-22-dev \
    libavresample-dev

RUN python3 -m pip install numpy cmake

RUN apt-get update && apt-get install -y libyaml-cpp-dev
RUN apt-get update && apt-get install -y wget

RUN usermod -a -G video ubuntu
USER ubuntu
