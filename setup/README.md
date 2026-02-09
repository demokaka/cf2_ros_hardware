## Clone the repository
```
git clone --recursive <this_repo>
```
## In case we forgot the --recursive, run the follwing command to download all the submodules
```
git submodule update --init --recursive 
```

# Usage:
## Build and run the container using `make` command
To run the docker container for simulation, use this command:
- On linux:
  - If there is nvidia gpu: `make create_con`
  - If not: `make create_con OPTION=linux_nogpu`

- On windows:
  - If there is nvidia gpu: `make create_con OPTION=windows`
  - If not: `make create_con OPTION=windows_nogp`

If it is the first time running the container, it will take a few minutes(depending on the internet) to download and build a docker image and then run a docker container. 
> **Note:**
> 
> The name of the docker image and container can be modify as specified in the Makefile with the name `IMAGE_NAME` and `CONTAINER_NAME`. 


## Open new terminal
We can access the container from another terminal either by `docker` command or by using `make`:
- Using docker command: `docker exec -it ros_gz_test_cont_01 bash`
- Using make: `make access_con`

## Using ros2 and Gazebo commands
Inside the docker container, to use the ROS and Gazebo commands, we need to source the entrypoint file using this command: `source /ros_entrypoint.sh`

Examples: 
- [ros2 tutorials](https://docs.ros.org/en/jazzy/Tutorials.html)
- [gazebo turtorials](https://gazebosim.org/docs/latest/tutorials/) 


### Editors
The source code can be edited in the `./src/ros_ws/src` folder using any editors available. It is recommended to use VSCode.
Then, 
- to upload the code to the container: `make update_con_src`
- to sync the data inside the container with the source code: `m̀ake sync_src`

