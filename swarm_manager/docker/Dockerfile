FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release git python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb [arch=amd64,arm64,armhf] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2-latest.list

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-base python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=jazzy
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

RUN pip install --no-cache-dir torch torchvision torchaudio numpy gym tqdm --break-system-packages

WORKDIR /workspace 
