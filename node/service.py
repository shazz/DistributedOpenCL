from rpyc.utils.server import ThreadedServer
from rpyopencl_service import RPyOpenCLService

RPYC_CFG = {"allow_all_attrs": True, "allow_pickle": True, "allow_public_attrs": True}

if __name__ == "__main__":
    s = ThreadedServer(RPyOpenCLService, port=18861, protocol_config=RPYC_CFG)
    s.start()
