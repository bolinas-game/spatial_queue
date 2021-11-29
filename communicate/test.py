from __future__ import print_function

import logging

import grpc
from communicate.server import move_pb2
from communicate.server import move_pb2_grpc

def create_point_request() -> move_pb2.StreamRequest:
    request = move_pb2.StreamRequest()
    request.se_info.x1 = 525399
    request.se_info.y1 = 4194928
    request.se_info.x2 = 525678
    request.se_info.y2 = 4194998
    yield request

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('35.233.239.245:80') as channel:
        stub = move_pb2_grpc.MoveStub(channel)

        response = stub.GetPosition(create_point_request())
        for i in response:
            print("client received: " + str(i))
        print("test complete")

if __name__ == '__main__':
    logging.basicConfig()
    run()

