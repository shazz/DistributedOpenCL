import pyopencl as cl
import numpy as np

import rpyc
from rpyc import Service
from RPyOpenCLContext import RPyOpenCLContext
import logging
from typing import Callable, Any

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(module)s - %(message)s')

class RPyOpenCLService(Service):

    contexts = {}

    device_types_mapping = {
        "{}".format(1 << 0): "Default",
        "{}".format(1 << 1): "CPU",
        "{}".format(1 << 2): "GPU",
        "{}".format(1 << 3): "ACCELERATOR",
        "{}".format(1 << 4): "CUSTOM",
    }     

    def _get_context(self, context_id):
        
        logging.debug("get Context {} from {}".format(context_id, RPyOpenCLService.contexts))

        if context_id not in RPyOpenCLService.contexts:
            raise RuntimeError("Unknown context id")
        return RPyOpenCLService.contexts[context_id]        

    def exposed_get_platforms(self):
        platforms = cl.get_platforms()

        platforms_desc = {}

        for index, platform in enumerate(platforms):
            platforms_desc[index] = {
                'name': platform.name,
                'profile': platform.profile,
                'vendor': platform.vendor,
                'version': platform.version,
                'devices': ["{} ({})".format(device.name, RPyOpenCLService.device_types_mapping[str(device.type)]) for device in platform.get_devices()]
            }  
        return platforms_desc

    def exposed_get_devices(self, platform_idx):
        devices = cl.get_platforms()[platform_idx].get_devices()

        devices_desc = {}

        for index, device in enumerate(devices):
            devices_desc[index] = {
                'name': device.name,
                'profile': device.profile,
                'vendor': device.vendor,
                'vendor_id': hex(device.vendor_id),
                'version': device.version,
                'type': RPyOpenCLService.device_types_mapping[str(device.type)],
                'driver_version': device.driver_version,
                'endian_little': device.endian_little,
                'extensions': device.extensions,
                'global_mem_size': device.local_mem_size,
                'local_mem_size': device.local_mem_size,
                'opencl_c_version': device.opencl_c_version,
                'max_clock_frequency': device.max_clock_frequency,
                'max_compute_units': device.max_compute_units,
                'image_support': True if device.image_support == 1 else False,
            }  

        return devices_desc        

    def exposed_create_context(self, platform_idx: int, device_idx: int) -> str:

        context = RPyOpenCLContext(platform_idx, device_idx)
        RPyOpenCLService.contexts[context.context_id] = context

        return context.context_id

    def exposed_add_command_queue(self, context_id: str) -> None:

        self._get_context(context_id).create_queue()

    def exposed_get_device_info(self, context_id: str) -> dict:

        return self._get_context(context_id).get_device_info()

    def exposed_create_input_buffer(self, context_id: str, local_object: Any) -> None:

        local_object = rpyc.classic.obtain(local_object)
        logging.debug("cast np to remove netref: {}".format(type(local_object)))
        if type(local_object) == np.ndarray:
            local_object = np.array(local_object)

        self._get_context(context_id).create_input_buffer(local_object)

    def exposed_create_output_buffer(self, context_id: str, object_type, shape) -> None:
        self._get_context(context_id).create_output_buffer(object_type, shape)        
    
    def exposed_compile_kernel(self, context_id: str, kernel: str, use_prefered_vector_size: str = None) -> None:
        self._get_context(context_id).compile_kernel(kernel, use_prefered_vector_size)

    def exposed_execute_kernel(self, context_id: str, kernel_name: str, work_size: tuple, wait_execution: bool = True) -> np.array:
        return self._get_context(context_id).execute_kernel(kernel_name, work_size, wait_execution)

    def exposed_run_perf_tests(self, context_id: str) -> None:
        self._get_context(context_id).run_perf_tests()

    def exposed_delete_context(self, context_id: str) -> None:
        if context_id in RPyOpenCLService.contexts:
            logging.debug("Removing context {}".format(context_id))
            del RPyOpenCLService.contexts[context_id]

    def exposed_set_callback(self, context_id: str, cb: Callable) -> None:
        logging.debug("Received remote callback: {}, {}".format(cb, type(cb)))
        self._get_context(context_id).set_callback(cb)       