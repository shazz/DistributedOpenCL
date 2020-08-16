import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpyopencl import RPyOpenCLCluster
from decorators import timer
import json
import numpy as np

# Globals to simplify sample tuning
object_type = np.float32
size = 50000
kernel_name = "sum_mul"
a_np = np.random.rand(size).astype(object_type)
b_np = np.random.rand(size).astype(object_type)
nodes = [
    {"name": "rpi-opencl1", "ip": "localhost"},
    {"name": "rpi-opencl2", "ip": "localhost"}
]


@timer
def compute_on_cluster(cl_context, kernel):

    print("Compiling kernel on all the nodes of the cluster")
    cl_context.compile_kernel(kernel, use_prefered_vector_size="float")

    print("Create 2 inputs buffers of size {} and type {}".format(type(a_np), a_np.shape))
    cl_context.create_input_buffer(local_object=a_np)
    cl_context.create_input_buffer(local_object=b_np)

    print("Create 2 output buffers of size {} and type {}".format(type(a_np), a_np.shape))
    cl_context.create_output_buffer(object_type=object_type, object_shape=a_np.shape)
    cl_context.create_output_buffer(object_type=object_type, object_shape=a_np.shape)

    print("Executing the Kernel")
    res_cl_np_arrays = np.array(cl_context.execute_kernel(kernel_name, (size, )))

    # cluster.disconnect_cluster_context(cl_context)
    # cluster.delete_cluster_context(cl_context)

    return res_cl_np_arrays


def compare_results(res_cl: np.array) -> None:
    """[summary]

    Args:
        res_cl ([type]): [description]
    """
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

    with open("kernels/{}.cl".format(kernel_name), "r") as kernel_file:
        kernel = kernel_file.read()

    print("Create Cluster")
    cluster = RPyOpenCLCluster(nodes, use_async=False)

    print("Get Platforms on the cluster")
    platforms = cluster.get_platforms()
    for platform in platforms:
        print(json.dumps(platform, indent=4, sort_keys=True))

    print("Creating a clustered context")
    context = cluster.create_cluster_context()

    print("Computing kernel {} on the clustered context".format(kernel_name))
    res_cl_cluster = compute_on_cluster(context, kernel)

    print("comparing cluster results")
    compare_results(res_cl_cluster)

    print("Cleaning")
    cluster.delete_cluster_context(context)
    cluster.disconnect_cluster_context(context)
