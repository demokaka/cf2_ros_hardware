IMAGE_NAME="cf2_ros_img:1.0.0"
CONTAINER_NAME="cf2_ros_img_cont_01"
IMAGE_NAME_BL="cf2_bl_ros_img:1.0.0"
CONTAINER_NAME_BL="cf2_bl_ros_img_cont_01"

# ENVIRONMENT += -e "GZ_SIM_RESOURCE_PATH="/root/ros_ws/src/multidrones_ws/multidrones_test/models/gazebo/""
# Define the folder where all sub-Makefiles are stored
SUB_MAKEFILES_DIR := sub_makefiles

# Default option if none is provided
DEFAULT_OPTION := linux

# Use the 'option' variable if provided, otherwise default to DEFAULT_OPTION
OPTION ?= $(DEFAULT_OPTION)

# Dynamically include the corresponding Makefile if it exists
ifneq ("$(wildcard $(SUB_MAKEFILES_DIR)/$(OPTION).mk)","")
    include $(SUB_MAKEFILES_DIR)/$(OPTION).mk
else
    $(error Makefile for option '$(OPTION)' not found in $(SUB_MAKEFILES_DIR))
endif


## Manage source code
SRC_DIR = ./src

create_src:
	mkdir -p ./src/ros_ws/src



sync_src:
	docker cp $(CONTAINER_NAME):/root/ros_ws/src/. $(SRC_DIR)/ros_ws/src

#	docker cp $(CONTAINER_NAME):/root/crazyflie-lib-python/. $(SRC_DIR)/crazyflie-lib-python
#	docker cp $(CONTAINER_NAME):/root/crazyflie-firmware/. $(SRC_DIR)/crazyflie-firmware

update_con_src:
	docker cp $(SRC_DIR)/ros_ws/src/. $(CONTAINER_NAME):/root/ros_ws/src


######################################################
##### Proxy management #####

install_jq:
	@if [ ! -f /usr/bin/jq ]; then\
		sudo apt install -y jq;\
	fi

enable_proxy: install_jq
	@echo "Enabling proxy configuration for Docker"

	@if [ -f /etc/docker/daemon.json ]; then \
		jq -s ".[0] * .[1]" /etc/docker/daemon.json proxy-config/daemon.json | sudo tee /etc/docker/daemon.json > /dev/null;\
	else \
		sudo cp proxy-config/daemon.json /etc/docker;\
	fi
	@sudo systemctl restart docker.service

	@if [ -f ~/.docker/config.json ]; then \
		jq -s ".[0] * .[1]" ~/.docker/config.json proxy-config/config.json > /tmp/config.json;\
		mv /tmp/config.json ~/.docker/config.json;\
	else \
		cp proxy-config/config.json ~/.docker;\
	fi

disable_proxy: install_jq
	@echo "Disabling proxy configuration for Docker"

	@if [ -f /etc/docker/daemon.json ]; then\
		jq "del(.proxies)" /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json > /dev/null;\
		sudo systemctl restart docker.service;\
	fi

	@if [ -f ~/.docker/config.json ]; then\
		jq "del(.proxies)" ~/.docker/config.json > /tmp/config.json;\
		mv /tmp/config.json ~/.docker/config.json;\
	fi

######################################################



######################################################
build_img:
	docker build --network host \
	-t $(IMAGE_NAME) -f ./Dockerfile ..

build_img_bl:
	docker build --network host \
	-t $(IMAGE_NAME_BL) -f ./Dockerfile.bl ..


##### Docker container #####	
# create docker container
# build_img
create_con: 
ifeq ($(filter linux linux_nogpu, $(OPTION)),)
	@docker run -it \
	--rm \
	--name $(CONTAINER_NAME) \
	$(ENVIRONMENT) \
	$(DEV) \
	$(NET) \
	$(GPU) \
	$(VOLUME) \
	$(IMAGE_NAME) \
	bash
else
	xhost +local:root
	@docker run -it \
	--rm \
	--name $(CONTAINER_NAME) \
	$(ENVIRONMENT) \
	$(DEV) \
	$(NET) \
	$(GPU) \
	$(VOLUME) \
	$(IMAGE_NAME) \
	bash
	
	xhost -local:root
endif

# build_img_bl
create_con_bl: 
ifeq ($(filter linux linux_nogpu, $(OPTION)),)
	@docker run -it \
	--rm \
	--name $(CONTAINER_NAME_BL) \
	$(ENVIRONMENT) \
	$(DEV) \
	$(NET) \
	$(GPU) \
	$(VOLUME) \
	$(IMAGE_NAME_BL) \
	bash
else
	xhost +local:root
	@docker run -it \
	--rm \
	--name $(CONTAINER_NAME_BL) \
	$(ENVIRONMENT) \
	$(DEV) \
	$(NET) \
	$(GPU) \
	$(VOLUME) \
	$(IMAGE_NAME_BL) \
	bash
	
	xhost -local:root
endif

access_con:
	docker exec -it $(CONTAINER_NAME) bash 

access_con_bl:
	docker exec -it $(CONTAINER_NAME_BL) bash 

##### Clean images/containers #####
# remove docker image
clean_img: 
	# IMAGE_ID=$(cat logs/image_id.txt) 
	# @docker rmi -f $(IMAGE_ID)
	docker rmi -f $(IMAGE_NAME)
	
clean_con: 
	@docker stop $(CONTAINER_NAME)
	@docker rm $(CONTAINER_NAME)