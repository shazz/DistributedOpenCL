import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rpyopencl import RPyOpenCLCluster
import json
import numpy as np
from PIL import Image
import logging
from decorators import timer

NODES = [ {"name": "rpi-opencl1", "ip": "localhost"}, {"name": "rpi-opencl2", "ip": "localhost"}, {"name": "rpi-opencl3", "ip": "localhost"}, {"name": "rpi-opencl4", "ip": "localhost"}    ]

@timer 
def setup_opencl_kernel():

    logging.debug("Create Cluster")
    cluster = RPyOpenCLCluster(NODES, use_async=True)

    logging.debug("Get Platforms on the cluster")
    platforms = cluster.get_platforms()
    for platform in platforms:
        logging.debug(json.dumps(platform, indent=4, sort_keys=True))

    logging.debug("Create context")
    print("Creating a clustered context")
    context = cluster.create_cluster_context()

    logging.debug("Reading and compiling the kernel using preferred vector size")
    with open("kernels/mandelbrot.cl", "r") as kernel_file:
        kernel = kernel_file.read()
        context.compile_kernel(kernel, use_prefered_vector_size=None)

    return cluster, context

@timer
def compute_fractal(cl_context, q, maxiter, is_update = False):

    array_type = np.uint16

    if not is_update:
        logging.debug("Create 2 inputs buffers")
        cl_context.create_input_buffer(q)
        cl_context.create_input_buffer(np.uint16(maxiter))

        logging.debug("Create 1 output buffer of type {} and shape {}".format(array_type, q.shape))
        cl_context.create_output_buffer(object_type=array_type, object_shape=q.shape)

    else:
        logging.debug("Updating maxiter buffer")
        cl_context.update_input_buffer(1, np.uint16(maxiter))

    logging.debug("Executing the kernel")
    res_np_arrays = np.array(cl_context.execute_kernel("mandelbrot", q.shape))

    return res_np_arrays

def generate_inputs(x1, x2, y1, y2, w, h):
    # draw the Mandelbrot set, from numpy example
    xx = np.arange(x1, x2, (x2-x1)/w)
    yy = np.arange(y2, y1, (y1-y2)/h) * 1j
    q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex64)

    return q

if __name__ == '__main__':

    # set width and height of window, each node will take care of a 1024x1024 sub window
    w = 512*len(NODES)
    h = 512*len(NODES)

    # setup remote OpenCL cluster context
    cluster, context = setup_opencl_kernel()

    # generate inputs
    zoom_factor = 0.5
    maxiter_start = 30
    maxiter_stop = 1000
    maxiter_step = 50
    q = generate_inputs(x1=-2.13*zoom_factor, x2=0.77*zoom_factor, y1=-1.3*zoom_factor, y2=1.3*zoom_factor, w=w, h=h)
    
    # Compute fractal
    for maxiter in range(maxiter_start, maxiter_stop, maxiter_step):
        output = compute_fractal(context, q, maxiter = maxiter, is_update=(maxiter != maxiter_start))

        logging.debug("Reshaping from {} to {}".format(output.shape, (w, h)))
        mandel = (output.reshape((h, w)) / float(output.max()) * 255.).astype(np.uint8)

        im = Image.fromarray(mandel)
        im.putpalette([i for rgb in ((j, 0, 0) for j in range(255))
                            for i in rgb])
        im.save("fractals/fractal_cluster_{}.png".format(maxiter))

    cluster.delete_cluster_context(context)
    cluster.disconnect_cluster_context(context)





