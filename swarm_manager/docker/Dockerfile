FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


RUN apt update && apt install -y \
    curl gnupg2 lsb-release git python3-pip python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb [arch=amd64,arm64,armhf] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2-latest.list


RUN apt update && apt install -y ros-jazzy-desktop python3-argcomplete && \
    rm -rf /var/lib/apt/lists/*


ENV ROS_DISTRO jazzy
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]


RUN pip install torch torchvision torchaudio numpy gym tqdm


WORKDIR /workspace
