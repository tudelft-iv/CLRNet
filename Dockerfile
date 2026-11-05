FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
#set shell 
SHELL ["/bin/bash", "-c"]
#set colors
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan
#start with root user
USER root

#NVIDIA GL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils && \
    rm -rf /var/lib/apt/lists/*

###Create new user
#
#Creating a user inside the container, so we won't work as root.
#Setting all setting all the groups and stuff.
#
###

#expect build-time argument
ARG HOST_USER_GROUP_ARG
#create group appuser with id 999
#create group hostgroup with ID from host. This is needed so appuser can manipulate the host files without sudo.
#create appuser user with id 999 with home; bash as shell; and in the appuser group
#change password of appuser to admin so that we can sudo inside the container
#add appuser to sudo, hostgroup and all default groups
#copy default bashrc and add ROS sourcing
RUN groupadd -g 999 appuser && \
    groupadd -g $HOST_USER_GROUP_ARG hostgroup && \
    useradd --create-home --shell /bin/bash -u 999 -g appuser appuser && \
    echo 'appuser:admin' | chpasswd && \
    usermod -aG sudo,hostgroup,plugdev,video,adm,cdrom,dip,dialout appuser && \
    cp /etc/skel/.bashrc /home/appuser/  

#basic dependencies for everything
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    netbase\
    git\
    build-essential\    
    wget\
    curl\
    gdb

#install ROS
#dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    lsb-release
RUN rm -rf /var/lib/apt/lists/*
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    ros-noetic-desktop-full
#install ROS tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    python3-rosdep\
    python3-rosinstall\
    python3-rosinstall-generator\
    python3-catkin-tools\
    python3-wstool
RUN rosdep init
RUN rosdep update

#install useful plugins and stuff
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    libgoogle-glog-dev\
    ros-noetic-pcl-ros\
    ros-noetic-octomap-ros\
    ros-noetic-octomap-msgs\
    ros-noetic-dynamixel-sdk\
    ros-noetic-effort-controllers\
    ros-noetic-jsk-rviz-plugins\
    ros-noetic-jsk-rqt-plugins\
    ros-noetic-joint-trajectory-controller\
    ros-noetic-usb-cam\
    ros-noetic-joy\
    ros-noetic-pointgrey-camera-driver\
    ros-noetic-octomap-rviz-plugins

#install projects
#dependencies
USER root

# Install Open3D system dependencies and pip
USER root
RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure appuser has proper ownership and permissions
RUN mkdir -p /home/appuser/src/calibration
RUN chown -R appuser:appuser /home/appuser/src

USER appuser

# create folder and copy the source files to install them
COPY --chown=appuser:appuser ./calibration /home/appuser/src/calibration/

USER root

# Ensure the CLRNet directory exists and has the correct ownership
RUN mkdir -p /home/appuser/src/calibration/CLRNet
RUN chown -R appuser:appuser /home/appuser/src/calibration/CLRNet

# Ensure the checkpoints directory exists
USER root
RUN mkdir -p /home/appuser/data/checkpoints && \
    chown -R appuser:appuser /home/appuser/data

# Download the DepthAnythingV2 model checkpoint (outside /home/appuser/data, which run.sh mounts over)
USER root
RUN mkdir -p /home/appuser/models && chown appuser:appuser /home/appuser/models
USER appuser
RUN wget -O /home/appuser/models/depth_anything_v2_vitl.pth \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true
USER appuser

# install requirements
# RUN cd /home/appuser/src/calibration/CLRNet && \
#     pip install -r requirements.txt

# Install DepthAnythingV2 requirements (the code itself is the git submodule under calibration/CLRNet)
USER root
RUN pip install -r /home/appuser/src/calibration/CLRNet/Depth-Anything-V2/requirements.txt

USER appuser