#!/usr/bin/bash
set -e
rm -rf build install log #Ensure packages are rebuilt to reflect changes
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source ./install/setup.bash