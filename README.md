# DistributedOpenCL
A PoC of distributed OpenCL on a Raspberry Pi 3 cluster

### Requirements
A cluster of n Raspberry Pi 3 running each :
- [Raspberry Pi OS (32-bit) Lite](https://www.raspberrypi.org/downloads/raspberry-pi-os/)
- [VC4CL](https://github.com/doe300/VC4CL)
- `libatlas-base-dev` (`sudo apt-get install libatlas-base-dev`)
- `pip3` (`sudo apt-get install python3-pip`)

### Installation
On each node of the cluster, run:
- copy the `node` folder
- `sudo pip3 install pyopencl rpyc`

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
