import socket
import sys

PORT_PYCHARM = 4242  # this should be forwarded to the correct port through ssh
PORT = int(sys.argv[1])


def forward_port(port):
    import subprocess
    subprocess.Popen("ssh woody -NL {}:127.0.0.1:{}".format(port, port).split())
    import time
    time.sleep(3)
    # os.system("netstat -tulpn | grep {}".format(port))


def main():
    print("Starting proxy...")
    debug = False
    if debug:
        f = open('proxy_log.txt', 'w')
        sys.stdout = f
        print("Started proxy.")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to the ssh port forwarding

    # create an INET, STREAMing socket
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind the socket to a public host, and a well-known port
    serversocket.bind(("", PORT))
    # become a server socket
    serversocket.listen(1)
    if debug:
        print("Listening on port {}".format(PORT), flush=True)

    # No while True as we will only accept one connection
    conn, addr = serversocket.accept()
    with conn:
        if debug:
            print("Connected by", addr, flush=True)

        s.connect(("127.0.0.1", PORT_PYCHARM))

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
