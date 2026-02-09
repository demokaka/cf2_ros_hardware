# How to setup the simulation environment
The setup is tested for 2 operating system(OS): 
- Linux(Ubuntu, Linux Mint,...)
- Windows

The simulation runs very well on the Linux machine. For Windows user, an installation of wsl machine is required.

Check if there are dedicated graphics on the machine. It is preferable to have a nvidia gpu for this setup. 
Otherwise, the simulation can be very slow(low timing factor and FPS) due to graphics rendering of robot requires high computation resources.
If there is one, please install the graphic driver for it. 

## For Linux user: The setup is described for Ubuntu
### Step 0: Install GPU driver
Install the proprietary Nvidia driver by opening the `Additional Drivers` software. If the driver is already installed, this step can be skipped.  

> :warning: **Note:**
> Do not download and install the nvidia driver from source. For each distribution, there will be available a way to install the nvidia driver on linux.
>

### Step 1: Install Docker 
- Install Docker for the current distro follow this [link](https://docs.docker.com/engine/install/)
- Do not forget to follow post-installation steps at this [link](https://docs.docker.com/engine/install/linux-postinstall/)

### Step 2: Install nvidia-container-toolkit(If you have nvidia gpu)
Follow this [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install and setup the necessary packages.
The detailed installation guide for Ubuntu distro is shown here:
- Configure the repository:  
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
- Optionally, configure the repository to use experimental packages:
```
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
- Update the packages list from the repository:
```
sudo apt-get update
```
- Install the NVIDIA Container Toolkit packages:
```
sudo apt-get install -y nvidia-container-toolkit
```
- Configure the container runtime by using the `nvidia-ctk` command:
```
sudo nvidia-ctk runtime configure --runtime=docker
```
- Restart the Docker daemon:
```
sudo systemctl restart docker
```
To configure the container runtime for Docker running in Rootless mode, follow these steps:
- Configure the container runtime by using the `nvidia-ctk` command:
```
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
```
- Restart the Rootless Docker daemon:
```
systemctl --user restart docker
```
- Configure `/etc/nvidia-container-runtime/config.toml` by using the `sudo nvidia-ctk` command:
```
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
```



## For Windows user
### Step 0: Install GPU driver
Download and install the [Nvidia driver](https://www.nvidia.com/Download/index.aspx). If the driver is already installed, this step can be skipped.  

> :warning: **Note:**
> 
> This is the only driver you need to install. Do not install any Linux display driver in WSL.



### Step 1: Install WSL
Launch your preferred Windows Terminal / Command Prompt / Powershell and install WSL:
```
wsl --install
```
Ensure you have the latest WSL kernel:
```
wsl --update
```
Install a distribution, an ubuntu distro is prefered for this setup:
```
wsl --install -d ubuntu
```
Some useful commands when using WSL:
- To list installed distribution: `wsl -l -v`
  To list available distribution for installation: `wsl -l -o`
- To run a wsl distribution: `wsl -d <DistributionName>`. For example: `wsl -d ubuntu`
- To set default distribution: `wsl -s <DistributionName>`. For example: `wsl -s ubuntu`. Now running `wsl` will run the default distribution.

> **Note:**
> 
> After install the distro and run it for the first time, user needs to wait for the installation, authentication. 
> An update step is very useful on the first run to get the latest update on the machine. For example, for Ubuntu distro, `sudo apt update && sudo apt upgrade-y`

### Step 2: Install Docker in the WSL machine
- Within the wsl machine, install Docker for the current distro follow this [link](https://docs.docker.com/engine/install/)
- Do not forget to follow post-installation steps at this [link](https://docs.docker.com/engine/install/linux-postinstall/)

> **Note:**
> 
> Install the Docker within the WSL machine. Do not install Docker Desktop application on Windows. 

### Step 3: Install nvidia-container-toolkit(If you have nvidia gpu)
Within the wsl machine, follow step 2 as for Linux machine

### Step 4: Install a X-Server software to manage the display
Go download and install [VcXsrv Windows X Server ](https://sourceforge.net/projects/vcxsrv/). When you want to run simulation with windows, it is needed to run this software, click next till finish. 