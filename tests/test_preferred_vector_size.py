import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../node')))

from rpyopencl import RPyOpenCLCluster
import numpy as np
from unittest import TestCase

from rpyc.utils.server import OneShotServer
from rpyopencl_service import RPyOpenCLService

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
    __global const float4 *a_g, __global const float4 *b_g, 
    __global float4 *res_add, __global float4 *res_mul)
{
  int gid = get_global_id(0);
  res_add[gid] = a_g[gid] + b_g[gid];
  res_mul[gid] = a_g[gid] * b_g[gid];
}
"""


class CommonTest(TestCase):

    def setUp(self):
        RPYC_CFG = {"allow_all_attrs": True, "allow_pickle": True, "allow_public_attrs": True}
        self.server = OneShotServer(RPyOpenCLService, port=18861, auto_register=False, protocol_config=RPYC_CFG)
        self.server.logger.quiet = False
        self.server._start_in_thread()

        print("OneShotServer started!")

        nodes = [ {"name": "pytest", "ip": "localhost"} ]
        cluster = RPyOpenCLCluster(nodes, use_async=False)
        self.node = cluster.get_node("pytest")

        print("Opencl node ready!")

    def tearDown(self):
        self.server.close()

class TestNoUseWithoutTemplate(CommonTest):

    @pytest.fixture(autouse=True)
    def setup_opencl(self):

        self.object_type = np.float32
        self.size = 16
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_no_template

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size=None)
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size,), True))

        print("Difference:", res_array[0] - (self.a_np + self.b_np))
        assert np.allclose(res_array[0], self.a_np + self.b_np)
        assert np.allclose(res_array[1], self.a_np * self.b_np)            

class TestNoUseWithTemplate(CommonTest):

    @pytest.fixture(autouse=True)
    def setup_opencl(self):

        self.object_type = np.float32
        self.size = 16
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_template

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size=None)
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size,), True))

        print("Difference:", res_array[0] - (self.a_np + self.b_np))
        assert np.allclose(res_array[0], self.a_np + self.b_np)
        assert np.allclose(res_array[1], self.a_np * self.b_np)      

class TestNoUseWithoutTemplateNoDivisible(CommonTest):

    @pytest.fixture(autouse=True)
    def setup_opencl(self):

        self.object_type = np.float32
        self.size = 17
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_no_template

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size=None)
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        # with pytest.raises(RuntimeError):
        res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size,), True))

        print("Difference:", res_array[0] - (self.a_np + self.b_np))
        assert np.allclose(res_array[0], self.a_np + self.b_np)
        assert np.allclose(res_array[1], self.a_np * self.b_np)   

class TestNoUseWithTemplateNoDivisible(CommonTest):

    @pytest.fixture(autouse=True)
    def setup_opencl(self):

        self.object_type = np.float32
        self.size = 17
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_template

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size=None)
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        # with pytest.raises(RuntimeError):
        res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size,), True))

        print("Difference:", res_array[0] - (self.a_np + self.b_np))
        assert np.allclose(res_array[0], self.a_np + self.b_np)
        assert np.allclose(res_array[1], self.a_np * self.b_np)     

class TestUseWithTemplateNoDivisible(CommonTest):

    @pytest.fixture(autouse=True)
    def setup_opencl(self):

        self.object_type = np.float32
        self.size = 17
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_template

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size="float")
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        with pytest.raises(RuntimeError):
            res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size,), True))

            print("Difference:", res_array[0] - (self.a_np + self.b_np))
            assert np.allclose(res_array[0], self.a_np + self.b_np)
            assert np.allclose(res_array[1], self.a_np * self.b_np)    

class TestUseWithoutTemplateNoDivisible(CommonTest):

    @pytest.fixture(autouse=True)
    def setup_opencl(self):

        self.object_type = np.float32
        self.size = 17
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_template

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size="float")
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        with pytest.raises(RuntimeError):
            res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size,), True))

            print("Difference:", res_array[0] - (self.a_np + self.b_np))
            assert np.allclose(res_array[0], self.a_np + self.b_np)
            assert np.allclose(res_array[1], self.a_np * self.b_np)    

class TestUseOptimizedNoDivisible(CommonTest):

    # WARNING!!!! This test will not pass even as the result is wrong! kernel conception issue!
    # There is no easy way to spot this conception issue unless checking the kernel input type

    @pytest.fixture(autouse=True)
    def setup_opencl(self):

        self.object_type = np.float32
        self.size = 17
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_no_template_optimized

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size="float")
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size // 4,), True))

        print("Difference:", res_array[0] - (self.a_np + self.b_np))
        assert not np.allclose(res_array[0], self.a_np + self.b_np)
        assert not np.allclose(res_array[1], self.a_np * self.b_np)   

class TestNoUseOptimized(CommonTest):

    @pytest.fixture(autouse=True)
    def setup_opencl_node_optimized(self):

        self.object_type = np.float32
        self.size = 16
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_no_template_optimized

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size=None)
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size // 4,), True))

        print("Difference:", res_array[0] - (self.a_np + self.b_np))
        assert np.allclose(res_array[0], self.a_np + self.b_np)
        assert np.allclose(res_array[1], self.a_np * self.b_np)     
      
class TestNoUseOptimizedNoDivide(CommonTest):

    # WARNING!!!! This test will not pass even as the result is wrong! kernel conception issue!
    # There is no easy way to spot this conception issue unless checking the kernel input type

    @pytest.fixture(autouse=True)
    def setup_opencl(self):

        self.object_type = np.float32
        self.size = 17
        self.kernel_name = "sum_mul"
        self.a_np = np.random.rand(self.size).astype(self.object_type)
        self.b_np = np.random.rand(self.size).astype(self.object_type)
        self.kernel = kernel_no_template_optimized

    def test_preferred_vector_size(self):

        self.node.create_context()
        self.node.add_command_queue()
        self.node.compile_kernel(self.kernel, use_prefered_vector_size=None)
        self.node.create_input_buffer(self.a_np)
        self.node.create_input_buffer(self.b_np)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)
        self.node.create_output_buffer(object_type=self.object_type, object_shape=self.a_np.shape)

        res_array = np.array(self.node.execute_kernel(self.kernel_name, (self.size // 4,), True))

        print("Difference:", res_array[0] - (self.a_np + self.b_np))
        assert not np.allclose(res_array[0], self.a_np + self.b_np)
        assert not np.allclose(res_array[1], self.a_np * self.b_np)          

