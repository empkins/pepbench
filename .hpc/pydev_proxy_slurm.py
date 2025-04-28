import socket
import sys

# NOTE: The main difference between this file and pydev_proxy.py is that the Slurm version runs on woody whereas the
# Torque version runs on the compute node (reason: we don't have the permission to do a local port forwarding from the
# compute node to woody when using Slurm - so we run the proxy on woody and setup the port forwarding from there)

PORT = int(sys.argv[1])
NODE = sys.argv[2]


def forward_port(port, local_port=None):
    import subprocess
    if not local_port:
        local_port = port
    subprocess.Popen(f"ssh -NR {port}:127.0.0.1:{local_port} {NODE}".split())
    import time
    time.sleep(3)


def main():
    print("Starting proxy...")
    debug = True
    if debug:
        f = open('proxy_log.txt', 'w')
        sys.stdout = f
        print("Started proxy.")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to the ssh port forwarding

    # create an INET, STREAMing socket
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind the socket to a public host, and a well-known port
    serversocket.bind(("", 0))
    local_port = serversocket.getsockname()[1]
    # setup connection from node to woody
    forward_port(PORT, local_port)
    # become a server socket
    serversocket.listen(1)
    if debug:
        print("Listening on port {}".format(PORT), flush=True)

    # No while True as we will only accept one connection
    conn, addr = serversocket.accept()
    with conn:
        if debug:
            print("Connected by", addr, flush=True)

        # connect to PyCharm (through PyCharm SSH port forwarding)
        s.connect(("127.0.0.1", PORT))

        if debug:
            print("Connected to PyCharm", flush=True)

        while True:
            data_pycharm = s.recv(1024)
            cmd = data_pycharm.decode("utf-8").strip().split("\t")
            if debug:
                print("< {}".format(data_pycharm), flush=True)

            if cmd[0] == "99":  # connect to given port
                port = int(cmd[2])
                if debug:
                    print("Forwarding port {}".format(port), flush=True)
                forward_port(port)

            conn.sendall(data_pycharm)
            data_dbg = conn.recv(1024)
            if debug:
                print("> {}".format(data_dbg), flush=True)
            s.sendall(data_dbg)

            if debug:
                f.flush()


if __name__ == '__main__':
    main()
