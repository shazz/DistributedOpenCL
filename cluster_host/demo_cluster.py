import numpy as np
from rpyopencl import OpenCLCluster
import time
import rpyc

# generate data on the cluster host

# OpenCL float16 is a vector of 16 float32, NOT a numpy float16 (half)
vector_size = 1000000
vector_type = np.float32

print("Generate {} random numbers of type {}".format(vector_size, str(vector_type)))
a_np = np.random.rand(vector_size).astype(vector_type)
b_np = np.random.rand(vector_size).astype(vector_type)
res_np = np.empty_like(a_np).astype(vector_type)
print("Buffer types: {}, {}, {} of size: {} bytes".format(a_np.dtype, b_np.dtype, res_np.dtype, a_np.__sizeof__()))

# create cluster
cl_cluster = OpenCLCluster(nodes = [ {"name": "rpi1", "ip": "localhost"} ])
# cl_cluster = OpenCLCluster(nodes = [ {"name": "rpi1", "ip": "localhost"}, {"name": "rpi2", "ip": "localhost"} ])

# get platforms for each cluster
print("Available platforms on the Clusters:")
for cluster_node_name in cl_cluster.get_nodes():
    platforms = cl_cluster.get_platforms(cluster_node_name)
    print("Platforms for node: {}".format(cluster_node_name))

    for index, platform in enumerate(platforms):
        print("{}\tPlatform: {}".format(index, platform))
        print("\tName: {}".format(platform.name))
        print("\tProfile: {}".format(platform.profile))
        print("\tVendor: {}".format(platform.vendor))
        print("\tVersion: {}".format(platform.version))

# split the data per nb of platforms
list_a_np = np.array_split(a_np, len(platforms))
list_b_np = np.array_split(b_np, len(platforms))
list_res_np = np.array_split(res_np, len(platforms))

cl_nodes = {}
for index, cluster_node_name in enumerate(cl_cluster.get_nodes()):

    # create openCL context on platform rpi1, first device
    print("Getting node for platform")
    node = cl_cluster.get_node(cluster_node_name)
    cl_nodes[cluster_node_name] = { 'node': node }
    platform = node.cl.get_platforms()

    device_nb = 0
    print("Create OpenCL context on device {}".format(device_nb))
    cl = node.cl
    ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platform[device_nb])]
    )
    device = ctx.devices[0]
    float_vector_size = device.preferred_vector_width_float

    print("Device {} properties:".format(device))
    print("\tPrefered float vector size: {}".format(float_vector_size))
    print("\tVersion: {}".format(device.version))
    print("\tVendor: {}".format(device.vendor_id))
    print("\tProfile: {}".format(device.profile))
    print("\topencl_c_version: {}".format(device.opencl_c_version))
    print("\tmax_compute_units: {}".format(device.max_compute_units))
    print("\tmax_clock_frequency: {}".format(device.max_clock_frequency))
    print("\tlocal_mem_size: {}".format(device.local_mem_size))
    print("\tglobal_mem_size: {}".format(device.global_mem_size))
    #print("\textensions: {}".format(device.extensions))

    print("Create OpenCL queue")
    queue = cl.CommandQueue(ctx)

    print("Create remote np split arrays and copy data there")
    t_trans0 = time.process_time_ns()
    s_a_np = node.np.array(list_a_np[index])
    s_b_np = node.np.array(list_b_np[index])
    s_res_np = node.np.array(list_res_np[index])
    t_trans1 = time.process_time_ns()

    print("Copy data to device buffers")
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=s_b_np)
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

    print("Reading kernel file: demo_float{}.cl".format(float_vector_size))
    with open("demo_float{}.cl".format(float_vector_size), "r") as f_kernel:
        kernel = f_kernel.read()

    print("Compiling kernel")
    prg = cl.Program(ctx, kernel).build()

    print("Executing computation")
    #prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
    sum_knl = prg.sum
    sum_knl.set_args(a_g, b_g, res_g)
    local_work_size = None
    #local_work_size = (10,)

    t0 = time.process_time_ns()
    ev = cl.enqueue_nd_range_kernel(queue=queue, kernel=sum_knl, global_work_size=(vector_size//float_vector_size,), local_work_size=local_work_size)
    ev.wait()
    t1 = time.process_time_ns()

    print("Transferring result to host")
    t2 = time.process_time_ns()
    cl.enqueue_copy(queue, s_res_np, res_g)
    t3 = time.process_time_ns()

    print("Updating local result")
    t_trans2 = time.process_time_ns()
    list_res_np[index] = np.array(s_res_np) #rpyc.classic.obtain(s_res_np)
    t_trans3 = time.process_time_ns()

    cl_nodes[cluster_node_name]['transfer_size'] = list_a_np[index].__sizeof__() + list_b_np[index].__sizeof__() + list_res_np[index].__sizeof__()
    cl_nodes[cluster_node_name]['time_transfer_cluster_to_host'] = (t_trans1-t_trans0)/1000000
    cl_nodes[cluster_node_name]['time_transfer_host_to_cluster'] = (t_trans3-t_trans2)/1000000
    cl_nodes[cluster_node_name]['time_copy_device_to_host'] = (t3-t2)/1000000
    cl_nodes[cluster_node_name]['time_compute'] = (t1-t0)/1000000

# Check on CPU with Numpy:
print("Computing on the host using numpy")
t4 = time.process_time_ns()
res_local = a_np + b_np
t5 = time.process_time_ns()
print("Local type:", res_local.dtype)

for array in list_res_np:
    print("Size of computed arrays: {}".format(len(array)))
print("Size of global array: {}".format(len(res_np)))

# concatenate all sub results
res_np = np.concatenate(list_res_np, axis=None)

print("Comparing results")
print("Difference :{}".format(res_np - res_local))
print(s_a_np[0:5])
print(s_b_np[0:5])
print(res_np[0:5])
print(res_local[0:5])

print("Checking the norm between both: {}".format(np.linalg.norm(res_np - res_local)))
print("Checking results are mostly the same: ", np.allclose(res_np, res_local))

print("Time to process:")
for node, data in cl_nodes.items():
    print("time to copy: {} ms for node {}".format(data['time_copy_device_to_host'], node))
    print("time to compute: {} ms for node {}".format(data['time_compute'], node))
    print("time to transfer to host: {} ms for node {} ({} MB/s)".format(data['time_transfer_cluster_to_host'], node,   round( (data['transfer_size']/(1024*1024))/(data['time_transfer_cluster_to_host']/1000), 2) ))
    print("time to transfer from host: {} ms for node {} ({} MB/s)".format(data['time_transfer_host_to_cluster'], node, round( (data['transfer_size']/(1024*1024))/(data['time_transfer_host_to_cluster']/1000), 2) ))

print("time to transfer using cluster: {} ms".format(sum([data['time_transfer_cluster_to_host'] + data['time_transfer_host_to_cluster'] for data in cl_nodes.values()])))
print("time to copy from device using cluster: {} ms".format(sum([data['time_copy_device_to_host'] for data in cl_nodes.values()])))
print("time to compute using cluster: {} ms".format(sum([data['time_compute'] for data in cl_nodes.values()])))
print("time to compute using numpy: {} ms".format((t5-t4)/1000000))

