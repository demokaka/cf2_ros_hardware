# IMAGE_NAME="multidrones_img:1.0.0"
# CONTAINER_NAME="multidrones_cont_01"

# IMAGE_NAME_H="multidrones_humble:1.0.0"
# CONTAINER_NAME_H="multidrones_cont_humble_01"

# Provide X11 and auth for authorization 
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

# Configure the environment variables
ENVIRONMENT += -e DISPLAY 		# container can use the display of the host
ENVIRONMENT += -e "QT_X11_NO_MITSHM=1" 	# set the environmental flag to force Qt to show properly
ENVIRONMENT += -e "XAUTHORITY=${XAUTH}"

ENVIRONMENT += -e "PULSE_SERVER=${PULSE_SERVER}" # https://stackoverflow.com/questions/68310978/playing-sound-in-docker-container-on-wsl-in-windows-11
ENVIRONMENT += -e LD_LIBRARY_PATH=/usr/lib/wsl/lib #=> necessary for graphics 
# ENVIRONMENT += -e MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
ENVIRONMENT += -e MESA_LOADER_DRIVER_OVERRIDE=radeonsi

# ENVIRONMENT += -e "GZ_SIM_RESOURCE_PATH="/root/ros_ws/src/crazyws/crazyflie-simulation/simulator_files/gazebo/""

# Configure the devices to be used
DEV += --device /dev/dri --device /dev/dxg

# Configure the network
NET += --network host

# Volume 
VOLUME += -v $(XSOCK):$(XSOCK):rw
VOLUME += -v $(XAUTH):$(XAUTH):rw
VOLUME += -v /mnt/wslg/:/mnt/wslg/ 
VOLUME += -v /dev:/dev
VOLUME += -v /usr/lib/wsl:/usr/lib/wsl

#----- ROS folders ----
CACHE=./cache
VOLUME += -v $(CACHE)/ros_ws/src:/root/ros_ws/src:rw
VOLUME += -v $(CACHE)/ros_ws/install:/root/ros_ws/install:rw
VOLUME += -v $(CACHE)/ros_ws/build:/root/ros_ws/build:rw
VOLUME += -v $(CACHE)/ros_ws/log:/root/ros_ws/log:rw




# Config to allow container to use the GPU (for Gazebo)

# GPU += --security-opt=label=disable
#GPU += -e __NV_PRIME_RENDER_OFFLOAD=1
#GPU += -e __GLX_VENDOR_LIBRARY_NAME=nvidia
# GPU += -e NVIDIA_VISIBLE_DEVICES=all
# GPU += -e NVIDIA_DRIVER_CAPABILITIES=all
# GPU += -e XAUTHORITY=$(XAUTH)
# GPU += --runtime nvidia 
GPU += --privileged
# GPU += --gpus all



