# Ahold_product_detector
This repository implements a product detector for products in the Albert Heijn supermarket


### Install this ROS Package for Ubuntu

Before starting, make sure to have ROS Noetic installed: http://wiki.ros.org/noetic/Installation

After installation, initialize a workspace in order to build and run the ROS package. Do this by running the following commands:

```console
mkdir YOUR_WORKSPACE_NAME
cd YOUR_WORKSPACE_NAME
mkdir src
cd src
```
We want to have the ROS package inside the 'src' directory, so now that we are in here we can clone the repository:
```console
git clone git@github.com:stijnla/Ahold_product_detector.git
```
Now return to your workspace directory, source the ROS environment if you haven't done so, and build the package by running the following lines:
```console
cd ..
source /opt/ros/noetic/setup.bash
catkin build
source /devel/setup.bash
```
Now that we have setup the ROS package, it is time to setup the python virtual environment for the python dependencies. For simplicity, make sure that the location of your virtual environment is NOT in your workspace, so that catkin does not try to build your virtual environment. After choosing a location of desire, run the following commands to create and activate your virtual environment:

```console
python -m venv PATH_TO_YOUR_VIRTUAL_ENV
source PATH_TO_YOUR_VIRTUAL_ENV/bin/activate
```

Now we install the dependencies for python in this virtual environment so it does not interfere with any other projects:

```console
pip install numpy
pip install opencv-python
pip install ultralytics
pip install roslibpy
pip install scipy
```

If you want to rebuild the ROS package, you should first deactivate the python virtual environment, than clean the build and re-build the package. Run the following commands in the workspace directory to achieve this:

```console
deactivate
catkin clean
catkin build
```