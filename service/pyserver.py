import socket
import threading
from json_serializer import unpack_json, pack_json
from sev_interface import Operations
import glob
import cv2 as cv
import os


listening_thread_exit = False


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
                # 开启全新的通信线程
                t = threading.Thread(target=distribution(listener))
                t.start()


def distribution(listener):
    operations = Operations("../weights/mrcnn-epoch-85.h5", None)
    data_socket, addr = listener.accept()
    print('{} connected'.format(addr))
    while True:
        length = int(data_socket.recv(8))
        command_pkg = unpack_json(recv_pack(data_socket, length))

        if command_pkg['pkg_type'] == 'command':
            command = command_pkg['command']

            # TODO: 反射调用指定命令
            if hasattr(operations, command):
                method = getattr(operations, command)

                img_dir = r"C:\Users\zhiyuan\Desktop\temp\test"
                paths = glob.glob(os.path.join(img_dir, '*.jpg'))
                imgs = []
                for path in paths:
                    img = cv.imread(path)
                    imgs.append(img)
                respond = method(imgs)
                # 发送信息
                length, data_pkg = pack_json(respond)
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


if __name__ == '__main__':
    respond_thread = RespondThread(max_link=10)
    respond_thread.start()

