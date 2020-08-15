import rpyc
import numpy as np
import logging
from typing import Callable, Any
from enum import Enum

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s - %(message)s')
RPYC_CFG = {"allow_all_attrs": True, "allow_pickle": True, "allow_public_attrs": True}

class DeviceType(Enum):
    CPU = "CPU"
    GPU = 1
    ACCELERATOR = 2
    ALL = 3

class ClusterContext():

    def __init__(self, nodes: dict, devices_to_use: DeviceType = DeviceType.CPU, use_async : bool = False):
        """[summary]

        Args:
            nodes (dict): [description]
            devices_to_use (DeviceType, optional): [description]. Defaults to DeviceType.ALL.
            use_async (bool, optional): [description]. Defaults to False.

        Raises:
            ValueError: [description]
        """
        self.context_nodes = nodes
        self.output_buffers_nb = 0
        self.use_async = use_async
        self.contexts = {}
        self.nb_ctx = 0

        # get available plarform for each node
        for node in self.context_nodes.values():
            logging.debug("looking for plaforms for node: {}".format(node.node_name))
            self.contexts[node.node_name] = []

            logging.debug("Adding all platform with devices of type: {} on node {}".format(devices_to_use, node.node_name))

            try:
                platforms = node.get_platforms()['platforms']
                for idx, platform in platforms.items():

                    for d_idx, device in enumerate(platform['devices']):
                        if devices_to_use == DeviceType.ALL or device['type'] == devices_to_use.value:
                            logging.debug("Adding device {} of type {}".format(device, device['type']))    
                            ctx = node.create_context(platform_idx = idx, device_idx = d_idx)
                            node.add_command_queue(ctx)
            
                            self.contexts[node.node_name].append(ctx)
                            logging.debug("Adding context {} for node {}: {}".format(ctx, node.node_name, self.contexts))
                        else:
                            logging.debug("Device {} of type {} discarded".format(device, device['type']))   
                
            except RuntimeError as e:
                logging.warning("Forgetting node: {} due to: {}".format(node, e))

        logging.debug("Contexts created: {}".format(self.contexts))
        
        # check we created at least one context
        
        for contexes in self.contexts.values():
            self.nb_ctx += len(contexes)
        if self.nb_ctx == 0:
            raise ValueError("No suitable platforms ({}) found to create a Cluster".format(devices_to_use))


    def compile_kernel(self, kernel: str, use_prefered_vector_size: str = None):
        """[summary]

        Args:
            kernel (str): [description]
            use_prefered_vector_size (str, optional): [description]. Defaults to None.
        """
        for node in self.context_nodes.values():
            for context in self.contexts[node.node_name]:
                logging.debug("Compiling kernel for node {} and context {}".format(node.node_name, context))
                node.compile_kernel(context, kernel, use_prefered_vector_size)

    def create_input_buffer(self, local_object: Any):
        """[summary]

        Args:
            local_object (Any): [description]
        """
        #FIXME Duplicate work if there is multiple contexts (=platforms) for one node
        if type(local_object) is np.ndarray:
            logging.debug("Splitting input array")
            split_array = np.array_split(local_object, len(self.context_nodes))
        
            if not self.use_async:
                for node, sub_array in zip(self.context_nodes.values(), split_array):
                    logging.debug("Create buffer with sub-array on node {}".format(node.node_name))
                    for context in self.contexts[node.node_name]:
                        node.create_input_buffer(context, sub_array)
            else:
                statuses = []

                for node, sub_array in zip(self.context_nodes.values(), split_array):
                    logging.debug("Create buffer with sub-array on node {}".format(node.node_name))
                    for context in self.contexts[node.node_name]:
                        statuses.append(node.create_input_buffer(context, sub_array))
            
                logging.debug("{} statuses are waiting to finish".format(len(statuses)))
                # waiting for all buffers to be created
                while len(statuses) > 0:
                    for status in statuses:
                        if status.ready:
                            statuses.remove(status)
                            logging.debug("One input buffer created, remaining: {}".format(len(statuses)))
        else:
            #FIXME: is it ok to send asynchronously the scalar buffer without waiting....
            logging.debug("No split for this input of type {}".format(type(local_object)))
            for node in self.context_nodes.values():
                for context in self.contexts[node.node_name]:
                    node.create_input_buffer(context, local_object)

    def create_output_buffer(self, object_shape: tuple, object_type: type):
        """[summary]

        Args:
            object_shape (tuple): [description]
            object_type (type): [description]
        """
        #FIXME: how to split on non 1D arrays ?
        split_array_shape = object_shape[0] // len(self.context_nodes)

        for node in self.context_nodes.values():
            for context in self.contexts[node.node_name]:
                node.create_output_buffer(context, split_array_shape, object_type)
        
        self.output_buffers_nb += 1
    
    def execute_kernel(self, kernel_name : str, work_size: int) -> np.array:
        """[summary]

        Args:
            kernel_name (str): [description]
            work_size (int): [description]

        Returns:
            np.array: [description]
        """
        #FIXME: divider should be self.nb_ctx
        divider = len(self.context_nodes)
        context_work_size = tuple(dim//divider for dim in work_size)
        # context_work_size = work_size // divider)
        logging.debug("worksize: {}, nodes: {}, context_work_size: {}".format(work_size, divider, context_work_size))
        logging.debug("output buffers: {}".format(self.output_buffers_nb))
        
        res_per_buffer = []
        res = []
        for output in range(self.output_buffers_nb):
            res_per_buffer.append([])

        if not self.use_async:
            for node in self.context_nodes.values():
                for context in self.contexts[node.node_name]:
                    sub_res = node.execute_kernel(context, kernel_name, context_work_size, wait_execution=True)
                    for output in range(self.output_buffers_nb):
                        output_res = sub_res[output]
                        logging.debug("Got sub result from node {} on output {} of size {}".format(node.node_name, output, output_res.shape))
                        res_per_buffer[output].append(output_res)
        else:
            statuses = {}
            results = {}
            for node in self.context_nodes.values():
                for context in self.contexts[node.node_name]:                
                    statuses[node.node_name] = node.execute_kernel(context, kernel_name, context_work_size, wait_execution=True)
            
            logging.debug("{} kernels are now wating to be completed".format(len(statuses)))

            #FIXME, there is no guarantee the order will be kept so, this code doesn't work as it is
            while statuses:
                for node in statuses.keys():
                    status = statuses[node]
                    if status.ready:
                        sub_res = np.array(status.value)
                        results[node] = { }
                        for output in range(self.output_buffers_nb):
                            results[node][output] = sub_res[output]

                        statuses = dict((k,v) for k,v in statuses.items() if k!=node)
                        logging.debug("One kernel completed, remaining: {}".format(len(statuses)))

            for output in range(self.output_buffers_nb):
                res_per_buffer[output] = [results[node][output] for node in sorted(results.keys())]


        logging.debug("Got all results, aggregating them")
        for output in range(self.output_buffers_nb):
            logging.debug("Concatenate results of shape {} for output {}".format(len(res_per_buffer[output]), output))
            res_agg = np.concatenate(res_per_buffer[output], axis=None)
            logging.debug("aggregated shape: {}".format(res_agg.shape))
            res.append(res_agg)

        return res

    def disconnect(self) -> None:
        """[summary]
        """

        for node in self.context_nodes.values():
            logging.debug("Disconneting node {}".format(node.node_name))
            node.disconnect()

    def _delete_contexts(self) -> None:

        for node in self.context_nodes.values():
            for context in self.contexts[node.node_name]:            
                logging.debug("Deleting context {} on node {}".format(context, node.node_name))
                node.delete_context(context)

        # reset
        self.nb_ctx = 0
        self.contexts = {}

class Node():
    """[summary]
    """
    def __init__(self, ip, name, use_async=False):
        """[summary]

        Args:
            ip ([type]): [description]
            name ([type]): [description]
            use_async (bool, optional): [description]. Defaults to False.

        Raises:
            RuntimeError: [description]
        """
        try:
            logging.debug("Connecting to node {} at {}".format(name, ip))

            self._conn = rpyc.connect(ip, 18861, config=RPYC_CFG)
            self.cluster_cl = self._conn.root
            self.platforms = dict(self.cluster_cl.get_platforms())

        except Exception as e:
            logging.error("Cannot connect to node {} at {} due to {}".format(name, ip, e))
            raise RuntimeError(e)
        
        self.contexts = []
        self.node_name = name
        self.ip = ip
        self.use_async = use_async

    def get_platforms(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """        
        return {
            "node": self.node_name,
            "platforms": self.platforms
        }

    def get_devices(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        devices = {}
        for idx, platform in self.platforms.items():
            devices[platform['name']] = []
            for device in dict(self.cluster_cl.get_devices(idx)).values():
                devices[platform['name']].append(device)
        
        return devices
    
    def create_context(self, platform_idx: int = 0, device_idx: int = 0) -> str:
        """[summary]

        Args:
            platform_idx (int, optional): [description]. Defaults to 0.
            device_idx (int, optional): [description]. Defaults to 0.

        Returns:
            str: [description]
        """     
        ctx = self.cluster_cl.create_context(platform_idx=platform_idx, device_idx=device_idx)
        self.contexts.append(ctx)

        return ctx

    def get_device_info(self, context_id: str) -> dict:
        """[summary]

        Raises:
            ValueError: [description]

        Returns:
            dict: [description]
        """        
        if context_id not in self.contexts:
            raise ValueError("No context {} found!".format(context_id))

        return dict(self.cluster_cl.get_device_info(context_id))

    def add_command_queue(self, context_id: str) -> None:
        """[summary]

        Args:
            context_id (str): [description]

        Raises:
            ValueError: [description]
        """     
        if context_id not in self.contexts:
            raise ValueError("No context {} found!".format(context_id))

        self.cluster_cl.add_command_queue(context_id)

    def compile_kernel(self, context_id: str, kernel: str, use_prefered_vector_size: str = None) -> None:
        """[summary]

        Args:
            context_id (str): [description]
            kernel (str): [description]
            use_prefered_vector_size (str, optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
        """   
        if context_id not in self.contexts:
            raise ValueError("No context {} found!".format(context_id))

        self.cluster_cl.compile_kernel(context_id, kernel, use_prefered_vector_size=use_prefered_vector_size)

    def create_input_buffer(self, context_id: str, local_object: Any) -> Any:
        """[summary]

        Args:
            context_id (str): [description]
            local_object (Any): [description]

        Raises:
            ValueError: [description]

        Returns:
            Any: [description]
        """  
        if context_id not in self.contexts:
            raise ValueError("No context {} found!".format(context_id))

        if self.use_async:
            logging.debug("Creating buffers asynchronously")
            _create_input_buffer = rpyc.async_(self.cluster_cl.create_input_buffer)
            return _create_input_buffer(context_id, local_object)
        else:
            self.cluster_cl.create_input_buffer(context_id, local_object)     
            return None
        

    def create_output_buffer(self, context_id: str, object_shape: tuple, object_type: type) -> None:
        """[summary]

        Args:
            context_id (str): [description]
            object_shape (tuple): [description]
            object_type (type): [description]

        Raises:
            ValueError: [description]
        """    
        if context_id not in self.contexts:
            raise ValueError("No context {} found!".format(context_id))

        self.cluster_cl.create_output_buffer(context_id, object_type=object_type, shape=object_shape)

    def execute_kernel(self, context_id: str, kernel_name : str, work_size: tuple, wait_execution: bool = True) -> np.array:
        """[summary]

        Args:
            context_id (str): [description]
            kernel_name (str): [description]
            work_size (tuple): [description]
            wait_execution (bool, optional): [description]. Defaults to True.

        Raises:
            ValueError: [description]

        Returns:
            np.array: [description]
        """      
        if context_id not in self.contexts:
            raise ValueError("No context {} found!".format(context_id))

        if type(work_size) != tuple:
            raise ValueError("work_size has to be a tuple")

        if self.use_async:
            logging.debug("Executing asynchronously the kernel")
            _execute_kernel = rpyc.async_(self.cluster_cl.execute_kernel)
            return _execute_kernel(context_id, kernel_name, work_size, wait_execution)
        else:
            return np.array(self.cluster_cl.execute_kernel(context_id, kernel_name, work_size, wait_execution))

    def delete_context(self, context_id: str) -> None:
        """[summary]

        Args:
            context_id (str): [description]

        Raises:
            ValueError: [description]
        """
        if context_id not in self.contexts:
            raise ValueError("No context {} found!".format(context_id))

        self.cluster_cl.delete_context(context_id)

    def disconnect(self) -> None:
        """[summary]
        """        
        self._conn.close()

    def set_callback(self, context_id: str, cb: Callable) -> None:
        """[summary]

        Args:
            context_id (str): [description]
            cb (Callable): [description]

        Raises:
            ValueError: [description]
        """
        if context_id not in self.contexts:
            raise ValueError("No context {} found!".format(context_id))

        self.cluster_cl.set_callback(context_id, cb)

class RPyOpenCLCluster():

    # nodes = [ {"name": "rpi1", "ip": "localhost"} ]

    def __init__(self, nodes: list, use_async : bool = False):
        """[summary]

        Args:
            nodes ([type]): [description]
            use_async (bool, optional): [description]. Defaults to False.
        """        

        self.nodes = {}
        self.use_async = use_async
        for node in nodes:
            try:
                self.nodes[node['name']] = Node(ip=node['ip'], name=node['name'], use_async=use_async)
            except RuntimeError as e:
                logging.warning("Forgetting node: {} due to: {}".format(node, e))

    def get_platforms(self, node_name: str = None) -> list:
        """[summary]

        Args:
            node_name (str, optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            list: [description]
        """        
        if node_name: 
            if node_name in self.nodes:
                return self.nodes[node_name].get_platforms()
            else:
                raise ValueError('Host {} not found'.format(node_name))
        else:
            res = []
            for node in self.nodes.values():
                try:
                    res.append(node.get_platforms())
                except RuntimeError as e:
                    logging.warning("Forgetting node: {} due to: {}".format(node, e))
            return res

    def get_node(self, node_name):
        """[summary]

        Args:
            node_name ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if node_name in self.nodes:
            return self.nodes[node_name]
        else:
            raise ValueError('Host {} not found'.format(node_name))

    def get_nodes(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.nodes.keys()

    def create_cluster_context(self) -> ClusterContext:
        """[summary]

        Returns:
            ClusterContext: [description]
        """
        cluster_context = ClusterContext(self.nodes, use_async=self.use_async)
        return cluster_context

    def delete_cluster_context(self, cluster_context) -> None:
        """[summary]

        Args:
            cluster_context ([type]): [description]
        """
        cluster_context._delete_contexts()
        cluster_context = None

    def disconnect_cluster_context(self, cluster_context) -> None:
        """[summary]

        Args:
            cluster_context ([type]): [description]
        """
        cluster_context.disconnect()





