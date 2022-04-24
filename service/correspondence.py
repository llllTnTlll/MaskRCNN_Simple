import socket
import threading
import json


listening_thread_exit = False


# socket监听线程
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
                t = threading.Thread(target=send_pck(data_socket, addr))
                t.start()


def send_pck(data_socket, addr):
    """
    线程池Task
    根据收到的指示回复信息
    :param data_socket:
    :param addr:
    :return:
    """
    print('{} connected'.format(addr))
    task_exit = False
    while not task_exit:
        try:
            # 获取接收信息长度
            length = int(data_socket.recv(8))
            pck = unpack_json(recv_pack(data_socket, length))
            print(pck)
            # if info == b'camera1':
            #     data = data_pack(frame1, 100)
            #     # 先发送数据长度信息
            #     data_socket.send(str.encode(str(len(data)).ljust(32)))
            #     # 发送图片数据
            #     data_socket.send(data)
            #     print('camera1 send')

        except ConnectionResetError:
            task_exit = True
    print("{} disconnected".format(addr))


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
    pck = [{
        "pkg_type": "data",
        "img_data": data['img'],
        "result": data['result']
    }]
    str_json = json.dumps(pck)
    byte_json = str.encode(str_json)
    length = len(byte_json)
    return length, byte_json


def unpack_json(bytes_json):
    str_json = bytes.decode(bytes_json)
    pck = json.loads(str_json)
    return pck


if __name__ == '__main__':
    # d = dict(
    #     img='imgdata',
    #     result='result'
    # )
    # _, b = pack_json(d)
    # j = unpack_json(b)
    respond_thread = RespondThread(max_link=10)
    respond_thread.start()

