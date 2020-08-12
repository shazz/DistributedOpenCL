import rpyc
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(module)s - %(message)s')
RPYC_CFG = {"allow_all_attrs": True, "allow_pickle": True, "allow_public_attrs": True}

class ClusterContext():

    def __init__(self, nodes: dict):

        self.context_nodes = nodes
        self.output_buffers_nb = 0

        for node in self.context_nodes.values():
            try:
               node.create_context()
               node.add_command_queue()

            except RuntimeError as e:
                logging.warning("Forgetting node: {} due to: {}".format(node, e))

    def compile_kernel(self, kernel: str, use_prefered_vector_size: str = None):

        for node in self.context_nodes.values():
            node.compile_kernel(kernel, use_prefered_vector_size)

    def create_input_buffer(self, local_object: any):

        if type(local_object) is np.ndarray:
            logging.debug("Splitting input array")
            split_array = np.array_split(local_object, len(self.context_nodes))
        
            for node, sub_array in zip(self.context_nodes.values(), split_array):
                node.create_input_buffer(sub_array)
        else:
            logging.debug("No split for this input")
            node.create_input_buffer(local_object)

    def create_output_buffer(self, object_shape: tuple, object_type: type):

        #FIXME: how to split on non 1D arrays ?
        split_array_shape = object_shape[0] // len(self.context_nodes)

        for node in self.context_nodes.values():
            node.create_output_buffer(split_array_shape, object_type)
        
        self.output_buffers_nb += 1
    
    def execute_kernel(self, kernel_name : str, work_size: int) -> np.array:

        divider = len(self.context_nodes)
        context_work_size = tuple(dim//divider for dim in work_size)
        # context_work_size = work_size // divider)
        logging.debug("worksize: {}, nodes: {}, context_work_size: {}".format(work_size, divider, context_work_size))
        logging.debug("output buffers: {}".format(self.output_buffers_nb))

        #FIXME: how to // this execution and concatenate the results correctly?
        
        res_per_buffer = []
        res = []
        for output in range(self.output_buffers_nb):
            res_per_buffer.append([])

        for node in self.context_nodes.values():
            sub_res = node.execute_kernel(kernel_name, context_work_size, wait_execution=True)
            for output in range(self.output_buffers_nb):
                output_res = sub_res[output]
                logging.debug("Got sub result from node {} on output {} of size {}".format(node.node_name, output, output_res.shape))
                res_per_buffer[output].append(output_res)

        for output in range(self.output_buffers_nb):
            logging.debug("Concatenate results of shape {} for output {}".format(len(res_per_buffer[output]), output))
            res_agg = np.concatenate(res_per_buffer[output], axis=None)
            logging.debug("aggregated shape: {}".format(res_agg.shape))
            res.append(res_agg)

        return res

    def _delete_context(self) -> None:

        for node in self.context_nodes.values():
            node.delete_context()        

    def disconnect(self) -> None:

        for node in self.context_nodes.values():
            node.disconnect()


class Node():

    def __init__(self, ip, name):

        try:
            logging.debug("Connecting to node {} at {}".format(name, ip))
            self._conn = rpyc.connect(ip, 18861, config=RPYC_CFG)
            self.cluster_cl = self._conn.root
            self.platforms = dict(self.cluster_cl.get_platforms())

        except Exception as e:
            logging.error("Cannot connect to node {} at {} due to {}".format(name, ip, e))
            raise RuntimeError(e)
        
        self.context = None
        self.node_name = name

    def get_platforms(self) -> dict:
        return {
            "node": self.node_name,
            "platforms": self.platforms
        }

    def get_devices(self) -> dict:

        devices = {}
        for idx, platform in self.platforms.items():
            devices[platform['name']] = []
            for device in dict(self.cluster_cl.get_devices(idx)).values():
                devices[platform['name']].append(device)
        
        return devices
    
    def create_context(self) -> None:
        #FIXME: set indexes as needed
        self.context = self.cluster_cl.create_context(platform_idx=0, device_idx=0)

    def get_device_info(self) -> dict:
        if self.context is None:
            raise ValueError("No context created!")

        return dict(cluster_cl.get_device_info(self.context)) 

    def add_command_queue(self) -> None:
        if self.context is None:
            raise ValueError("No context created!")

        self.cluster_cl.add_command_queue(self.context)

    def compile_kernel(self, kernel: str, use_prefered_vector_size: str = None) -> None:
        if self.context is None:
            raise ValueError("No context created!")

        self.cluster_cl.compile_kernel(self.context, kernel, use_prefered_vector_size=use_prefered_vector_size)

    def create_input_buffer(self, local_object: any) -> None:
        if self.context is None:
            raise ValueError("No context created!")

        self.cluster_cl.create_input_buffer(self.context, local_object)

    def create_output_buffer(self, object_shape: tuple, object_type: type) -> None:
        if self.context is None:
            raise ValueError("No context created!")

        self.cluster_cl.create_output_buffer(self.context, object_type=object_type, shape=object_shape)
    
    def execute_kernel(self, kernel_name : str, work_size: tuple, wait_execution: bool = True) -> np.array:

        if type(work_size) != tuple:
            raise ValueError("work_size has to be a tuple")

        return np.array(self.cluster_cl.execute_kernel(self.context, kernel_name, work_size, wait_execution))

    def delete_context(self) -> None:
        self.cluster_cl.delete_context(self.context)

    def disconnect(self) -> None:
        self._conn.close()

class RPyOpenCLCluster():

    # nodes = [ {"name": "rpi1", "ip": "localhost"} ]

    def __init__(self, nodes: []):

        self.nodes = {}
        for node in nodes:
            try:
                self.nodes[node['name']] = Node(ip=node['ip'], name=node['name'])
            except RuntimeError as e:
                logging.warning("Forgetting node: {} due to: {}".format(node, e))

    def get_platforms(self, node_name: str = None) -> list:
        
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

        if node_name in self.nodes:
            return self.nodes[node_name]
        else:
            raise ValueError('Host {} not found'.format(node_name))

    def get_nodes(self):

        return self.nodes.keys()

    def create_cluster_context(self) -> ClusterContext:

        cluster_context = ClusterContext(self.nodes)
        return cluster_context

    def delete_cluster_context(self, cluster_context) -> None:

        cluster_context._delete_context()
        cluster_context = None



