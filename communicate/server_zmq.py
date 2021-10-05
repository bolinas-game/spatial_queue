#
#   Hello World communicate in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
import sys
sys.path.insert(0,"/Users/wangyanglan/Public/Project/spatial_queue")
from game import test_game
import global_var as gl
import threading
import logging

def get_logger(name):
    logging.basicConfig(filename='server.log', filemode='w',
                    format='%(asctime)s - %(message)s', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger(name).addHandler(console)
    return logging.getLogger(name)


def get_data(socket):

    while True:
        #  Wait for next request from client
        direction_message = socket.recv()
        # print("Received request: %s" % direction_message)

        #  Do some 'work'.
        #  Try reducing sleep time to 0.01 to see how blazingly fast it communicates
        #  In the real world usage, you just need to replace time.sleep() with
        #  whatever work you want python to do, maybe a machine learning task?
        # time.sleep(1)
        # print("sended")
        # s1 = 'send: '+message.decode()

        #  Send reply back to client
        #  In the real world usage, after you finish your work, send your output here

def send_data(socket):
    data, config, update_data = test_game.initialize()
    checked_links = []
    t = 0
    old_link_id = None
    points_df = test_game.extra_link_info(data)

    while True:
        get_message = socket.recv().decode("utf-8")
        if get_message!="test":
            gl.set_value("direction_message", get_message)


        print("Receive: " + get_message )
        logger.info("Receive: " + get_message )

        data, config, update_data, output, old_link_id, image_position = test_game.test_each(t, data, config, update_data, logger, old_link_id, points_df)

        t += 1
        if image_position==None:
            logger.info("calculated data: " +output)
            socket.send(str.encode(output))
        else:
            logger.info("calculated data: " +image_position + output)
            socket.send(str.encode(image_position+output))
        logger.info("send")

        # test_game.test_main()
        # while True:
        #     output = gl.get_value("output")
        #     # if output and not output.empty():
        #     if output and old_output!=output:
        #         # output_time = output.get()
        #         # output_time = output
        #         logger.info("calculated data: "+output)
        #         print(output)
        #
        #         socket.send(str.encode(output))
        #         logger.info("sended")
        #         old_output = output
        #         break



if __name__=="__main__":
    logger = get_logger('root')
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    gl._init()
    gl.set_value("direction_message", "test")
    send_data(socket)
    # test_game.test()
    # thread1 = threading.Thread(target=get_data, args=(socket,))
    # thread2 = threading.Thread(target=send_data, args=(socket,))
    # thread1.start()
    # thread2.start()
    # test_game.test_main()