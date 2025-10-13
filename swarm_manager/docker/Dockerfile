FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip \
    git curl wget build-essential \
    lsb-release gnupg2 software-properties-common \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-base \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    gz-harmonic \
    ros-jazzy-ros-gz \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

COPY src/ /workspace/src/
WORKDIR /workspace

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


RUN pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install numpy gymnasium stable-baselines3 tensorboard matplotlib

RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

RUN echo '#!/bin/bash\n\
source /opt/ros/jazzy/setup.bash\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
