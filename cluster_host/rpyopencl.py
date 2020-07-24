import rpyc

class Node():

    def __init__(self, ip, name):

        try:
            self._conn = rpyc.classic.connect(ip)
            self.cl = self._conn.modules.pyopencl
            self.np = self._conn.modules.numpy
            self.name = name
        except Exception as e:
            print("Cannot connect to node {} at {} due to {}".format(name, ip, e))
            raise RuntimeError(e)


class OpenCLCluster():

    # nodes = [ {"name": "rpi1", "ip": "localhost"} ]

    def __init__(self, nodes: []):

        self.nodes = {}
        for node in nodes:
            try:
                self.nodes[node['name']] = Node(ip=node['ip'], name=node['name'])
            except RuntimeError as e:
                print("Forgetting node: {} due to: {}".format(node, e))

    def get_platforms(self, node_name):
        
        if node_name in self.nodes:
            return self.nodes[node_name].cl.get_platforms()
        else:
            raise ValueError('Host {} not found'.format(node_name))


    def get_node(self, node_name):

        if node_name in self.nodes:
            return self.nodes[node_name]
        else:
            raise ValueError('Host {} not found'.format(node_name))

    def get_nodes(self):

        return self.nodes.keys()
