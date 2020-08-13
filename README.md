# DistributedOpenCL
A PoC of distributed OpenCL on a Raspberry Pi 3 cluster

### How to run the demos?

 1. Follow the requirements and configration below

 1. On one slave node of the cluster run:
 ```bash
 cd node
 sudo python3 service.py
 ```

 1. on the master node of the cluster run:
 ```bash
 cd samples
 python3 app_<SAMPLE_NAME>.py
 ```

### Requirements
A cluster of n Raspberry Pi 3 running each :
- [Raspberry Pi OS (32-bit) Lite](https://www.raspberrypi.org/downloads/raspberry-pi-os/)
- [VC4CL](https://github.com/doe300/VC4CL)
- optional [pocl](https://github.com/ogmacorp/pocl) to have a second OpenCL platform using the ARM CPU

### Configure the cluster nodes

for Raspberry PIs 3:

 1. Flash as Micro SD card using the latest Raspbian / Raspberry Pi OS

 2. Before installing the SD Card, do the following

 Enable Wifi
 ```
 touch ssh
 cat > wpa_supplicant.conf
 country=US
 ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
 update_config=1

 network={
  scan_ssid=1
  ssid="<your_ssid>"
  psk="<your_wifi_password>"
 }
 ```
 
 3. Set the Ethernet adapter with a static IP (ex: 10.0.0.5)
 ```
 sudo nano /etc/dhpcd.conf  
 # It is possible to fall back to a static IP if DHCP fails:
 # define static profile
 profile static_eth0
 static ip_address=10.0.0.5/24
 #static routers=192.168.1.1
 #static domain_name_servers=192.168.1.1

 # fallback to static profile on eth0
 interface eth0
 fallback static_eth0
 ```

 4. Install the SD card and boot

 5. from the master node, copy its SSH keys
 
 ```
 # copy rsa keys
 ssh-copy-id 10.0.0.5
 ```

### Installation
On each node of the cluster, run:
- copy the `node` folder
- `sudo apt-get install libatlas-base-dev`
- `sudo apt-get install python3-pip`
- `sudo pip3 install pyopencl rpyc`

### Start the nodes
On each node of the cluster, run:
- copy the `node` folder
- `cd node`
- `sudo python3 opencl_node.py`

### Start the cluster host
On any 3rd party computer or any node, run:
- copy the `cluster_host` folder
- `pip3 install rpyc numpy` if on a 3rd party computer
- `cd cluster_host`
- `python3 demo_cluster.py`

### How to install dependencies?

 1. How to install `pocl` on the rpi 3?
 
 ````
 sudo apt install git libhwloc-dev ocl-icd-opencl-dev ocl-icd-libopencl1 ocl-icd-dev  libhwloc-dev zlib1g zlib1g-dev clinfo libglew-dev zlib1g-dev libedit-dev libclang-7-dev git cmake llvm-7 clang-7
 #sudo apt install -y build-essential pkg-config libclang-dev ninja-build  dialog apt-utils
 git clone https://github.com/pocl/pocl.git
 cd pocl
 mkdir build; cd build
 cmake -DLLC_HOST_CPU=cortex-a53 -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-7 -DSTATIC_LLVM=1 -DENABLE_ICD=1 ..
 make
 sudo make install
 make check
 
 sudo cp /usr/local/etc/OpenCL/vendors/pocl.icd /etc/OpenCL/vendors/
 sudo clinfo
 ````
 
 2. How to install `VC4CL` on the rpi 3?
 
 ````
 # get third party software
 sudo apt-get install cmake git ocl-icd-opencl-dev ocl-icd-dev opencl-headers clinfo libraspberrypi-dev

 #get Clang compiler
 sudo apt-get install clang clang-format clang-tidy 

 git clone https://github.com/doe300/VC4CLStdLib.git
 git clone https://github.com/doe300/VC4C.git
 git clone https://github.com/doe300/VC4CL.git
 
 cd VC4CLStdLib
 mkdir build
 cd build/
 cmake ..
 make
 sudo make install
 sudo ldconfig
 
 cd ../../VC4C
 mkdir build
 cd build/
 cmake ..
 make
 sudo make install
 sudo ldconfig
 
 cd ../../VC4CL
 mkdir build
 cd build/
 cmake ..
 make
 sudo make install
 sudo ldconfig 
 
 sudo clinfo
 ````
