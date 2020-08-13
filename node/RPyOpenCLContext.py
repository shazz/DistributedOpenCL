import pyopencl as cl
import pyopencl.characterize.performance as perf

import numpy
import numpy as np

import rpyc
import uuid
import logging
from typing import Callable, Any
import traceback 

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s - %(message)s')

class RPyOpenCLContext():

    def __init__(self, platform_idx, device_idx):

        # generate a context id
        self.context_id = uuid.uuid4().hex

        logging.debug("Creating Context {} ".format(self.context_id))

        # keep it for future improbable reasons
        self.platform_idx = platform_idx
        self.device_id = device_idx

        # check indexes are available
        platforms = cl.get_platforms()
        if platform_idx > len(platforms) - 1:
            raise ValueError("No platform found at index {}".format(platform_idx))
        self.platform = platforms[platform_idx]

        devices = self.platform.get_devices()
        if device_idx > len(devices) - 1:
            raise ValueError("No device found at index {} on this platform".format(device_idx))        
        self.device = devices[device_idx]
        
        # variables to store program and queue
        self.prg = None
        self.queue = None

        # input OpenCL buffers
        self.input_buffers = []

        # output OpenCL buffers and associated numpy arrays
        self.output_buffers = []
        self.output_arrays = []

        logging.debug("Create OpenCL context for platform {}".format(self.platform))
        self.ctx = cl.Context(
            dev_type=cl.device_type.ALL,
            properties=[(cl.context_properties.PLATFORM, self.platform)]
        )

        # store prefered vector sizes
        self.preferred_vector_size = {
            "char": self.device.preferred_vector_width_char,
            "short": self.device.preferred_vector_width_short,
            "int": self.device.preferred_vector_width_int,
            "long": self.device.preferred_vector_width_long,
            "half": self.device.preferred_vector_width_half,
            "float": self.device.preferred_vector_width_float,
            "double": self.device.preferred_vector_width_double
        }

    def get_device_info(self):

        device_info = {
            'name': self.device.name,
            'prefered_vector_size': self.preferred_vector_size,
            'opencl_c_version': self.device.opencl_c_version,
            'max_compute_units': self.device.max_compute_units,
            'max_clock_frequency': self.device.max_clock_frequency,
            'local_mem_size': self.device.local_mem_size,
            'global_mem_size': self.device.global_mem_size,
            'extensions': self.device.extensions
        }

        return device_info

    def create_queue(self):

        self.queue = cl.CommandQueue(self.ctx)

    def run_perf_tests(self):

        if self.queue is not None:

            prof_overhead, latency = perf.get_profiling_overhead(self.ctx)
            logging.debug("command latency: %g s" % latency)
            logging.debug("profiling overhead: %g s -> %.1f %%" % (prof_overhead, 100*prof_overhead/latency))     

            logging.debug("empty kernel: %g s" % perf.get_empty_kernel_time(self.queue))
            logging.debug("float32 add: %g GOps/s" % (perf.get_add_rate(self.queue)/1e9))        

            for tx_type in [perf.HostToDeviceTransfer, perf.DeviceToHostTransfer, perf.DeviceToDeviceTransfer]:
                logging.debug("----------------------------------------")
                logging.debug(tx_type.__name__)
                logging.debug("----------------------------------------")

                logging.debug("latency: %g s" % perf.transfer_latency(self.queue, tx_type))
                for i in range(6, 30, 2):
                    bs = 1 << i
                    try:
                        result = "%g GB/s" % (perf.transfer_bandwidth(self.queue, tx_type, bs)/1e9)
                    except Exception as e:
                        result = "exception: %s" % e.__class__.__name__
                    logging.debug("bandwidth @ %d bytes: %s" % (bs, result))
        else:
            raise RuntimeError("perf tests cannot be executed without a queue")

    def create_input_buffer(self, local_object) -> None:

        if type(local_object) is numpy.ndarray:
            buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(local_object))
            self.input_buffers.append(buffer)
        else:
            logging.debug("Adding as scalar type: ", type(local_object))
            self.input_buffers.append(local_object)

    def create_output_buffer(self, object_type, shape):

        object_type = rpyc.classic.obtain(object_type)
        shape = rpyc.classic.obtain(shape)

        logging.debug("Creating output buffer of shape {}".format(shape))
        logging.debug("Creating output buffer of type {}".format(str(object_type)))

        destbuf = np.zeros(shape).astype(object_type)
        #logging.debug(destbuf.shape, type(destbuf), destbuf, destbuf.nbytes)

        buffer = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, destbuf.nbytes)
        self.output_buffers.append(buffer)
        self.output_arrays.append(destbuf)

    def compile_kernel(self, kernel : str, use_prefered_vector_size : str):
        if use_prefered_vector_size not in [None, 'char', 'short', 'int', 'long', 'half', 'float', 'double']:
            raise ValueError("Unknown vector size: {}".format(use_prefered_vector_size))
        
        if use_prefered_vector_size is not None:
            if '<VECSIZE>' not in kernel:
                logging.debug("No <VECSIZE> template found in kernel, preferred vector size won't be used")
                self.use_prefered_vector_size = None
            else:
                if self.preferred_vector_size[use_prefered_vector_size] != 0:
                    logging.debug("patching kernel to use prefered vector size")
                    kernel = kernel.replace('<VECSIZE>', str(self.preferred_vector_size[use_prefered_vector_size]))
                    logging.debug(kernel)
                    self.use_prefered_vector_size = use_prefered_vector_size
                else:
                    raise RuntimeError("This device doesn't support this type: {}".format(use_prefered_vector_size))

        self.prg = cl.Program(self.ctx, kernel)
        self.prg.build()

    def execute_kernel(self, kernel_name: str, work_size: tuple, wait_execution: bool = True):

        for kernel in self.prg.all_kernels():
            logging.debug("Kernel available: {}".format(kernel.get_info(cl.kernel_info.FUNCTION_NAME)))
            if kernel.get_info(cl.kernel_info.FUNCTION_NAME) == kernel_name:
    
                try:
                    if kernel.get_info(cl.kernel_info.NUM_ARGS) != len(self.input_buffers) + len(self.output_buffers):
                        raise ValueError("Kernel function args number ({}) is different from input and ouput buffers ({})".format(kernel.get_info(cl.kernel_info.NUM_ARGS), len(self.input_buffers)))

                    logging.debug("Setting args")
                    logging.debug(self.input_buffers)
                    logging.debug(self.output_buffers)

                    kernel.set_args(*self.input_buffers, *self.output_buffers)

                    local_work_size = None
                    divider = 1
                    if self.use_prefered_vector_size is not None:
                        divider = self.preferred_vector_size[self.use_prefered_vector_size]
                        
                    global_work_size = tuple(dim//divider for dim in work_size)
                    # global_work_size = work_size//divider

                    logging.debug("work_size: {}, global_work_size: {}, divider: {}".format(work_size, global_work_size, divider))
                    self.enqueue_event = cl.enqueue_nd_range_kernel(
                        queue=self.queue, 
                        kernel=kernel, 
                        global_work_size=global_work_size, 
                        local_work_size=local_work_size)

                    if wait_execution:
                        self.enqueue_event.wait()

                        logging.debug("enqueue from {} to {}".format(self.output_buffers, self.output_arrays))
                        for array, buffer in zip(self.output_arrays, self.output_buffers):
                            cl.enqueue_copy(self.queue, array, buffer)

                        logging.debug("Return local arrays: {}".format(self.output_arrays))
                        return self.output_arrays
                    else:
                        logging.debug("Adding callback on COMPLETE event")
                        self.enqueue_event.set_callback(cl.command_execution_status.COMPLETE, self.copy_on_callback)
                        return None
                        
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError("Unexpected error", e)

                # kernel found, done!
                break

        raise RuntimeError("Kernel {} not found".format(kernel_name))            

    def copy_on_callback(self, status):

        try:
            if self.callback is None:
                raise RuntimeError("Remote callback is not set!")

            logging.debug("On callback for event {}, enqueue from {} to {}".format(status, self.output_buffers, self.output_arrays)) 
            for array, buffer in zip(self.output_arrays, self.output_buffers):
                cl.enqueue_copy(self.queue, array, buffer)

            logging.debug("Call remote callback {} with results".format(type(self.callback)))
            self.callback(self.output_arrays)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("Unexpected error on callback", e)            

    def set_callback(self, cb):

        logging.debug("Callback set to {}".format(cb))
        self.callback = cb
