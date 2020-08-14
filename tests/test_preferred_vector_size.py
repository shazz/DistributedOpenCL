import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpyopencl import RPyOpenCLCluster
import numpy as np

kernel_template = """
__kernel void sum_mul(
    __global const float<VECSIZE> *a_g, __global const float<VECSIZE> *b_g, 
    __global float<VECSIZE>  *res_add, __global float<VECSIZE> *res_mul)
{
  int gid = get_global_id(0);
  res_add[gid] = a_g[gid] + b_g[gid];
  res_mul[gid] = a_g[gid] * b_g[gid];
}
"""

kernel_no_template = """
__kernel void sum_mul(
    __global const float *a_g, __global const float *b_g, 
    __global float *res_add, __global float *res_mul)
{
  int gid = get_global_id(0);
  res_add[gid] = a_g[gid] + b_g[gid];
  res_mul[gid] = a_g[gid] * b_g[gid];
}
"""

kernel_no_template_optimized = """
__kernel void sum_mul(
    __global const float8 *a_g, __global const float8 *b_g, 
    __global float8 *res_add, __global float8 *res_mul)
{
  int gid = get_global_id(0);
  res_add[gid] = a_g[gid] + b_g[gid];
  res_mul[gid] = a_g[gid] * b_g[gid];
}
"""

@pytest.fixture
def setup_opencl_node_no_template():

    nodes = [ {"name": "pytest", "ip": "localhost"} ]

    cluster = RPyOpenCLCluster(nodes, use_async=False)
    node = cluster.get_node("pytest")

    object_type = np.float32
    size = 16
    kernel_name = "sum_mul"
    a_np = np.random.rand(size).astype(object_type)
    b_np = np.random.rand(size).astype(object_type)

    return node, a_np, b_np, kernel_no_template, kernel_name, size, object_type

@pytest.fixture
def setup_opencl_node_template():

    nodes = [ {"name": "pytest", "ip": "localhost"} ]

    cluster = RPyOpenCLCluster(nodes, use_async=False)
    node = cluster.get_node("pytest")

    object_type = np.float32
    size = 16
    kernel_name = "sum_mul"
    a_np = np.random.rand(size).astype(object_type)
    b_np = np.random.rand(size).astype(object_type)

    return node, a_np, b_np, kernel_template, kernel_name, size, object_type

@pytest.fixture
def setup_opencl_node_template_no_int_divide():

    nodes = [ {"name": "pytest", "ip": "localhost"} ]

    cluster = RPyOpenCLCluster(nodes, use_async=False)
    node = cluster.get_node("pytest")

    object_type = np.float32
    size = 17
    kernel_name = "sum_mul"
    a_np = np.random.rand(size).astype(object_type)
    b_np = np.random.rand(size).astype(object_type)

    return node, a_np, b_np, kernel_template, kernel_name, size, object_type      

def test_no_use_without_template(setup_opencl_node_no_template):

    node, a_np, b_np, kernel, kernel_name, size, object_type = setup_opencl_node_no_template
    node.create_context()
    node.add_command_queue()
    node.compile_kernel(kernel, use_prefered_vector_size=None)
    node.create_input_buffer(a_np)
    node.create_input_buffer(b_np)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)

    res_array = np.array(node.execute_kernel(kernel_name, (size,), True))

    print(res_array)
    assert np.allclose(res_array[0], a_np + b_np)
    assert np.allclose(res_array[1], a_np * b_np)    

def test_no_use_with_template(setup_opencl_node_template):

    node, a_np, b_np, kernel, kernel_name, size, object_type = setup_opencl_node_template
    node.create_context()
    node.add_command_queue()
    node.compile_kernel(kernel, use_prefered_vector_size=None)
    node.create_input_buffer(a_np)
    node.create_input_buffer(b_np)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)

    res_array = np.array(node.execute_kernel(kernel_name, (size,), True))

    print("Difference:", res_array[0] - (a_np + b_np))
    assert np.allclose(res_array[0], a_np + b_np)
    assert np.allclose(res_array[1], a_np * b_np)

def test_no_use_with_template_no_int_divide(setup_opencl_node_template_no_int_divide):

    node, a_np, b_np, kernel, kernel_name, size, object_type = setup_opencl_node_template_no_int_divide
    node.create_context()
    node.add_command_queue()
    node.compile_kernel(kernel, use_prefered_vector_size=None)
    node.create_input_buffer(a_np)
    node.create_input_buffer(b_np)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)

    res_array = np.array(node.execute_kernel(kernel_name, (size,), True))

    print("Difference:", res_array[0] - (a_np + b_np))
    assert np.allclose(res_array[0], a_np + b_np)
    assert np.allclose(res_array[1], a_np * b_np)

def test_use_with_template(setup_opencl_node_template):

    node, a_np, b_np, kernel, kernel_name, size, object_type = setup_opencl_node_template
    node.create_context()
    node.add_command_queue()
    node.compile_kernel(kernel, use_prefered_vector_size="float")
    node.create_input_buffer(a_np)
    node.create_input_buffer(b_np)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)

    res_array = np.array(node.execute_kernel(kernel_name, (size,), True))

    print("Difference:", res_array[0] - (a_np + b_np))
    assert np.allclose(res_array[0], a_np + b_np)
    assert np.allclose(res_array[1], a_np * b_np)     

def test_use_with_template_no_int_divide(setup_opencl_node_template_no_int_divide):

    node, a_np, b_np, kernel, kernel_name, size, object_type = setup_opencl_node_template_no_int_divide
    node.create_context()
    node.add_command_queue()
    node.compile_kernel(kernel, use_prefered_vector_size="float")
    node.create_input_buffer(a_np)
    node.create_input_buffer(b_np)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)
    node.create_output_buffer(object_type=object_type, object_shape=a_np.shape)

    with pytest.raises(RuntimeError):
        res_array = np.array(node.execute_kernel(kernel_name, (size,), True))

        print("Difference:", res_array[0] - (a_np + b_np))
        assert np.allclose(res_array[0], a_np + b_np)
        assert np.allclose(res_array[1], a_np * b_np)     