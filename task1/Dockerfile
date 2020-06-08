FROM nvidia/cudagl:10.1-devel-ubuntu18.04

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libcudnn7=7.6.2.24-1+cuda10.0  \
#     libcudnn7-dev=7.6.2.24-1+cuda10.0

RUN apt-get update && apt-get install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3-pip
# RUN apt-get update && apt-get install -y python3.7

RUN apt-get update && apt-get install -y sudo locales

# Create user ubuntu and add it as a sudoer with no password
RUN adduser --disabled-password --shell /bin/bash --gecos "User" ubuntu && \
    sudo adduser ubuntu sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    sudo locale-gen en_US.UTF-8

RUN python3 -m pip install --upgrade cmake

RUN python3 -m pip install numpy opencv-python opencv-contrib-python
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

# RUN add-apt-repository ppa:savoury1/ffmpeg4
# RUN add-apt-repository ppa:savoury1/graphics
# RUN add-apt-repository ppa:savoury1/multimedia

RUN add-apt-repository ppa:jonathonf/ffmpeg-4
# RUN apt-get update && apt-get install -y ffmpeg
# RUN ffmpeg -version

RUN apt-get update && apt-get install -y libavcodec-dev libavformat-dev

USER ubuntu
