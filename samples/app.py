import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpyopencl import RPyOpenCLCluster
import json
import numpy as np
import time

import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.process_time_ns() 
        value = func(*args, **kwargs)
        end_time = time.process_time_ns()      
        run_time = end_time - start_time
        run_time = (end_time - start_time)/10e6
        print(f"Finished {func.__name__!r} in {run_time:.4f} ms")
        return value
    return wrapper_timer

# Globals to simplify sample tuning
object_type = np.float32
size = 50000
a_np = np.random.rand(size).astype(object_type)
b_np = np.random.rand(size).astype(object_type)
nodes = [ {"name": "rpi-opencl1", "ip": "localhost"}, {"name": "rpi-opencl2", "ip": "localhost"}  ]

@timer
def compute_on_one_node(node, kernel):

    print("Get Devices for node 1")
    node_devices = node.get_devices()
    for device in node_devices.values():
        print(json.dumps(device, indent=4, sort_keys=True))

    print("Create context")
    node.create_context()

    print("Add command queue on context")
    node.add_command_queue()

    node.compile_kernel(kernel, use_prefered_vector_size="float")

    print("Create 2 inputs buffers of shape {}".format(a_np.shape))
    node.create_input_buffer(a_np)
    node.create_input_buffer(b_np)

    print("Create 2 output buffers of size {} and type {}".format(object_type, a_np.shape))
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)

    print("Executing the Kernel")
    res_np_arrays = node.execute_kernel("sum", (size,), True)

    node.delete_context()

    return res_np_arrays

@timer
def compute_on_cluster(cluster, kernel):

    print("Creating a clustered context")
    cl_context = cluster.create_cluster_context()

    print("Compiling kernel on all the nodes of the cluster")
    cl_context.compile_kernel(kernel, use_prefered_vector_size = "float")

    print("Create 2 inputs buffers of size {} and type {}".format(type(a_np), a_np.shape))
    cl_context.create_input_buffer(local_object=a_np)
    cl_context.create_input_buffer(local_object=b_np)

    print("Create 2 output buffers of size {} and type {}".format(type(a_np), a_np.shape))
    cl_context.create_output_buffer(object_type=object_type, object_shape=a_np.shape)
    cl_context.create_output_buffer(object_type=object_type, object_shape=a_np.shape)

    print("Executing the Kernel")
    res_cl_np_arrays = cl_context.execute_kernel("sum", (size, ))

    return res_cl_np_arrays

def compare_results(res_cl):
    print("Comparing the (a+b) and (a*b) results with local numpy")
    res_sum_np = np.array(res_cl[0])
    print("Result sum", res_sum_np)
    print("Difference:", res_sum_np - (a_np + b_np))
    print("Norm", np.linalg.norm(res_sum_np - (a_np + b_np)))
    assert np.allclose(res_sum_np, a_np + b_np)

    res_mul_np = np.array(res_cl[1])
    print("Result mul", res_mul_np)
    print("Difference:", res_mul_np - (a_np * b_np))
    print("Norm", np.linalg.norm(res_mul_np - (a_np * b_np)))
    assert np.allclose(res_mul_np, a_np * b_np)

if __name__ == "__main__":

    print("Reading the kernel using preferred vector size")

    with open("kernels/sum_mul.cl", "r") as kernel_file:
        kernel = kernel_file.read()

    print("Create Cluster")
    cluster = RPyOpenCLCluster(nodes)

    print("Get Platforms on the cluster")
    platforms = cluster.get_platforms()
    for platform in platforms:
        print(json.dumps(platform, indent=4, sort_keys=True))

    print("Get Platforms for node 1")
    node1 = cluster.get_node("rpi-opencl1")
    node1_platforms = node1.get_platforms()
    for platform in node1_platforms.values():
        print(json.dumps(platform, indent=4, sort_keys=True))

    res_cl = compute_on_one_node(node1, kernel)
    compare_results(res_cl)

    res_cl = compute_on_cluster(cluster, kernel)
    compare_results(res_cl)

    node1.disconnect()

    cluster.delete_cluster_context(cl_context)
    cluster.disconnect()    

    