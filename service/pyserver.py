import socket
import threading
import json


listening_thread_exit = False
mydata = {
    "img_data": "img",
    "result": "result"
}


class RespondThread(threading.Thread):
    def __init__(self, max_link):
        super().__init__()
        self.max_link = max_link

    def run(self):
        host = '127.0.0.1'
        port = 50001
        # 使用IPV4和TCP协议创建socket监听
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((host, port))
            listener.listen()
            print('listener thread has begin...')
            while not listening_thread_exit:
                data_socket, addr = listener.accept()
                # 开启全新的通信线程
                t = threading.Thread(target=distribution(data_socket, addr))
                t.start()


def distribution(data_socket, addr):
    print('{} connected'.format(addr))
    length = int(data_socket.recv(8))
    command_pkg = unpack_json(recv_pack(data_socket, length))

    if command_pkg['pkg_type'] == 'command':
        while True:
            # TODO: 反射调用指定命令
            length, data_pkg = pack_json(mydata)
            try:
                data_socket.send(length)
                data_socket.send(data_pkg)
            except ConnectionResetError as e:
                print(repr(e))
                print('{} disconnected'.format(addr))
                return


def recv_pack(sock, count):
    """
    封装socket.recv
    保证能接收到指定长度信息
    :param sock:
    :param count:
    :return:
    """
    pack = b''
    while count:
        sentence = sock.recv(count)
        if not sentence:
            return None
        pack += sentence
        count -= len(sentence)
    return pack


def pack_json(data):
    pck = {
        "pkg_type": "data",
        "img_data": data['img_data'],
        "result": data['result']
    }
    str_json = json.dumps(pck)
    byte_json = str.encode(str_json)
    length = str(len(byte_json)).ljust(8)
    byte_length = str.encode(length)
    return byte_length, byte_json


def unpack_json(bytes_json):
    str_json = bytes.decode(bytes_json)
    pck = json.loads(str_json)
    return pck


if __name__ == '__main__':
    respond_thread = RespondThread(max_link=10)
    respond_thread.start()
