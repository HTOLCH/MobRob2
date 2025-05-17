ARG ROS_DISTRO=jazzy

FROM ros:jazzy-ros-base

WORKDIR /workspace

ENV ROS_DOMAIN_ID=7

ENV DISPLAY=:1

RUN apt-get update && \
    apt-get install -y vim \
    ros-jazzy-xacro \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-joint-state-publisher ros-jazzy-ros-gz-sim \
    ros-jazzy-slam-toolbox \
    ros-jazzy-tf2-ros \
    vim \
    xpdf \
    libusb-1.0-0-dev \
    python3-pip \
    ros-jazzy-tf-transformations \
    ros-jazzy-rqt-robot-steering \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-tf2-geometry-msgs \
    wget \
    dpkg \
    build-essential \
    cmake \
    ros-jazzy-joy-linux \
    #ros-jazzy-cv-bridge \
    #ros-jazzy-rclcpp \
    #ros-jazzy-sensor-msgs \
    #ros-jazzy-phidgets-drivers \
    #libphidget22 \
    libusb-1.0-0-dev \
    automake \
    python3-colcon-common-extensions \
    git \
    doxygen \
    python3-tk \
    python3-rosdep \
    python3-pip \
    #python3-opencv \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Clone and build AriaCoda
RUN git clone https://github.com/reedhedges/AriaCoda.git /workspace/src/AriaCoda \
     && cd /workspace/src/AriaCoda \
     && make \
     && make install 

#RUN git clone https://github.com/ros-drivers/phidgets_drivers.git /workspace/src/phidgets_drivers

#RUN git clone https://github.com/luxonis/depthai.git /workspace/src/depthai \
#    && cd /workspace/src/depthai 


#RUN pip install --break-system-packages depthai

#RUN sudo wget -qO- https://docs.luxonis.com/install_depthai.sh | bash
#RUN python3 -m pip install --break-system-packages depthai-viewer

RUN apt update && sudo apt install ros-jazzy-depthai-ros -y

# Copy the entire content of your local workspace into the Docker container
COPY . /workspace

# Copy the library to the system's library directory
#COPY ./libphidget22 /usr/local/lib


RUN pip install --break-system-packages phidget22
RUN pip install --break-system-packages blobconverter
RUN pip install --break-system-packages opencv-python
RUN pip install --break-system-packages depthai
RUN pip install --break-system-packages numpy
#RUN pip install --break-system-packages depthai

RUN apt-get update && apt-get install -y libgl1
RUN sudo apt-get install usbutils

RUN apt-get update && apt-get install -y libxrandr2 libxrender1 libxfixes3 libxcursor1 libxinerama1

RUN pip install --break-system-packages depthai-sdk
RUN pip install --break-system-packages pillow

RUN sudo apt update && sudo apt install -y ros-jazzy-pcl-conversions

RUN apt-get update && apt-get install -y \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rviz2 \
    ros-jazzy-nmea-navsat-driver

RUN pip install ultralytics --break-system-packages


#RUN git clone https://github.com/RichbeamTechnology/Lakibeam_ROS2_Driver.git src/Lakibeam_ROS2_Driver
    #&& cd /workspace/src/Lakibeam_ROS2_Driver \
    #&& colcon build --symlink-install


# Install ROS dependencies using rosdep
RUN rosdep update && \
    grep -F "source /opt/ros/${ROS_DISTRO}/setup.bash" /root/.bashrc || echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    grep -F "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" /root/.bashrc || echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /root/.bashrc

#RUN export XDG_RUNTIME_DIR=/tmp/runtime-root
#RUN mkdir -p $XDG_RUNTIME_DIR
#RUN chmod 700 $XDG_RUNTIME_DIR


# Build the ROS 2 workspace using colcon
#RUN . /opt/ros/jazzy/setup.sh && \
#   colcon build --symlink-install
    #rosdep install phidgets_drivers
    #--packages-ignore ariaNode#

# When the container starts, source ROS setup files and run bash
#CMD ["bash", "-c", "source /opt/ros/jazzy/setup.bash && colcon build && source /workspace/install/setup.bash && bash"]
    