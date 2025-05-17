### Gazebo Harmonic - ROS2 Jazzy

Build Dockerfile under the name tag `gazebo:dev`. If you want to change the name tag, please also change the image name inside `compose.yaml` file.
<br><br>

Before compose up, make sure to disable access control to display by input the following command:
```bash
xhost +
```

### Troubleshooting
If there is an error say cant find display: run command `export | grep DISPLAY` to find out if `DISPLAY` env existed. Else export `DISPLAY` accordingly.