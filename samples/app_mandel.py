import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpyopencl import RPyOpenCLCluster
import json
import numpy as np
from PIL import Image
import logging
import functools
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(module)s - %(message)s')

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.process_time_ns()  
        value = func(*args, **kwargs)
        end_time = time.process_time_ns()   
        run_time = (end_time - start_time)/10e6
        print(f"Finished {func.__name__!r} in {run_time:.4f} ms")
        return value
    return wrapper_timer


@timer 
def setup_opencl_kernel():

    logging.debug("Create Cluster")
    nodes = [ {"name": "rpi-opencl1", "ip": "localhost"} ]
    cluster = RPyOpenCLCluster(nodes)

    logging.debug("Get Platforms on the cluster")
    platforms = cluster.get_platforms()
    for platform in platforms:
        logging.debug(json.dumps(platform, indent=4, sort_keys=True))

    logging.debug("Get Platforms for node 1")
    node = cluster.get_node("rpi-opencl1")
    node_platforms = node.get_platforms()
    for platform in node_platforms.values():
        logging.debug(json.dumps(platform, indent=4, sort_keys=True))

    logging.debug("Get Devices for node 1")
    node_devices = node.get_devices()
    for device in node_devices.values():
        logging.debug(json.dumps(device, indent=4, sort_keys=True))

    logging.debug("Create context")
    node.create_context()

    logging.debug("Add command queue on context")
    node.add_command_queue()

    logging.debug("Reading and compiling the kernel using preferred vector size")
    with open("kernels/mandelbrot.cl", "r") as kernel_file:
        kernel = kernel_file.read()
        node.compile_kernel(kernel, use_prefered_vector_size="float")

    return node

@timer
def compute_fractal(node, q, maxiter):

    array_type = np.uint16

    logging.debug("Create 2 inputs buffers")
    node.create_input_buffer(q)
    node.create_input_buffer(np.uint16(maxiter))

    logging.debug("Create 1 output buffer of type {} and shape {}".format(array_type, q.shape))
    node.create_output_buffer(object_type=array_type, object_shape=q.shape)

    logging.debug("Executing the kernel")
    res_np_arrays = node.execute_kernel("mandelbrot", q.shape, True)

    return res_np_arrays

def generate_inputs(x1, x2, y1, y2, w, h):
    # draw the Mandelbrot set, from numpy example
    xx = np.arange(x1, x2, (x2-x1)/w)
    yy = np.arange(y2, y1, (y1-y2)/h) * 1j
    q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex64)

    return q

if __name__ == '__main__':

    # set width and height of window, more pixels take longer to calculate
    w = 2048
    h = 2048

    # setup remote OpenCL node
    node = setup_opencl_kernel()

    # generate inputs
    q = generate_inputs(x1=-2.13, x2=0.77, y1=-1.3, y2=1.3, w=w, h=h)
    
    # Compute fractal
    output = compute_fractal(node, q, maxiter = 30)

    logging.debug("Reshaping from {} to {}".format(output.shape, (w, h)))
    mandel = (output.reshape((h, w)) / float(output.max()) * 255.).astype(np.uint8)

    im = Image.fromarray(mandel)
    im.putpalette([i for rgb in ((j, 0, 0) for j in range(255))
                        for i in rgb])
    im.save("fractal.png")

    node.delete_context()
    node.disconnect()





