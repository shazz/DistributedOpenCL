# DistributedOpenCL
A PoC of distributed OpenCL on a Raspberry Pi 3 cluster

### Requirements
A cluster of n Raspberry Pi 3 running each :
- [Raspberry Pi OS (32-bit) Lite](https://www.raspberrypi.org/downloads/raspberry-pi-os/)
- [VC4CL](https://github.com/doe300/VC4CL)
- optional [pocl](https://github.com/ogmacorp/pocl)

### Installation
On each node of the cluster, run:
- copy the `node` folder
- `sudo apt-get install libatlas-base-dev`
- `sudo apt-get install python3-pip`
- `sudo pip3 install pyopencl rpyc`

Optional
- install `pocl` to have a second OpenCL platform using the ARM CPU

### Start the nodes
On each node of the cluster, run:
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
 sudo apt install libhwloc-dev ocl-icd-opencl-dev libglew-dev zlib1g-dev libedit-dev libclang-7-dev
 git clone https://github.com/ogmacorp/pocl.git
 cd pocl
 mkdir build; cd build
 cmake -DLLC_HOST_CPU=cortex-a53 -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-7 -DSTATIC_LLVM=1 -DENABLE_ICD=1 ..
 make
 sudo make install
 make check
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
