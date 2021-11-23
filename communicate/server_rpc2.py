from concurrent import futures
import logging
import sys
import os
import argparse
import grpc
from server import move_pb2
from server import move_pb2_grpc
from game import test_game
from threading import RLock

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_dir', metavar='N', type=str, default="../",
                    help='path for root_dir')

args = parser.parse_args()
root_directory = os.path.join(os.getcwd(),args.root_dir)
sys.path.insert(0, args.root_dir)

def get_logger(name):
    logging.basicConfig(filename='server.log', filemode='w',
                    format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(name)

    if not getattr(logger, 'handler_set', None):
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(console)
        logger.handler_set = True

    return logger

logger = get_logger('root')


def create_point_response(id, lon, lat, x, y, angle, t) -> move_pb2.StreamResponse:
    response = move_pb2.StreamResponse()
    response.point_info.id = id
    response.point_info.lon = lon
    response.point_info.lat = lat
    response.point_info.x = x
    response.point_info.y = y
    response.point_info.angle = angle
    response.point_info.t = t
    return response

def create_stop_response(isStop, t) -> move_pb2.StreamResponse:
    response = move_pb2.StreamResponse()
    response.stop_info.isStop = isStop
    response.stop_info.t = t
    return response

def create_image_response(image, t) -> move_pb2.StreamResponse:
    response = move_pb2.StreamResponse()
    response.change_info.image = image
    response.change_info.t = t
    return response

def create_complete_response(result, t) -> move_pb2.StreamResponse:
    response = move_pb2.StreamResponse()
    response.complete_info.complete = result
    response.complete_info.t = t
    return response

def create_newse_response(x, y) -> move_pb2.StreamResponse:
    response = move_pb2.StreamResponse()
    response.new_se_info.x1 = x[0]
    response.new_se_info.y1 = x[1]
    response.new_se_info.x2 = y[0]
    response.new_se_info.y2 = y[1]
    return response

class PeerSet(object):
    def __init__(self):
        self._peers_lock = RLock()
        self._peers = {}
        self.instances = {}

    def connect(self, peer, request):
        with self._peers_lock:
            if peer not in self._peers:
                logger.info("Peer {} connecting".format(peer))
                self._peers[peer] = 1
                move_instance = test_game.Trasport((request.se_info.x1, request.se_info.y1),
                                                        (request.se_info.x2, request.se_info.y2))
                self.instances[peer] = move_instance
                new_start, new_end = move_instance.get_new_od()
                return new_start, new_end
            else:
                self._peers[peer] += 1
                move_instance = self.instances[peer]
                output = self.update_each(move_instance,request.dir_info.op)
                return output

    def update_each(self,move_instance, direction_tmp):
        print("run update")
        whole_vehs, isStop, image_path,complete, t = move_instance.test_each_grpc(direction_tmp)
        while len(whole_vehs)==0:
            whole_vehs, isStop, image_path,complete, t = move_instance.test_each_grpc(direction_tmp)
            if len(whole_vehs)>0:
                break
        return whole_vehs, isStop, image_path, complete, t

    def disconnect(self, peer):
        logger.info("Peer {} disconnecting".format(peer))
        with self._peers_lock:
            if peer not in self._peers:
                raise RuntimeError("Tried to disconnect peer '{}' but it was never connected.".format(peer))
            del self._peers[peer]
            del self.instances[peer]

    def peers(self):
        with self._peers_lock:
            return self._peers.keys()


class Move(move_pb2_grpc.Move):
    def __init__(self):
        # self._dir_responded = threading.Event()
        self._peer_set = PeerSet()

    def _record_peer(self, context, request):
        return self._peer_set.connect(context.peer(), request)

    def GetPosition(self, request_iterator, context):
        # print(context.peer())
        request = next(request_iterator)
        if request.HasField("dir_info"):
            logger.info("receive direction {}".format(request.dir_info.op))
            # self._dir_responded.set()
            # whole_vehs, isStop, image_path, complete = self.update_each(direction_tmp = request.dir_info.op)
            whole_vehs, isStop, image_path, complete, t = self._record_peer(context, request)
            for index, row in whole_vehs.iterrows():
                r_veh_id, r_veh_coord_offset_x, r_veh_coord_offset_y, x2, y2, angle, t, link_id = int(
                    row["veh_id"]), row["lon_offset_utm"], row["lat_offset_utm"], row["lon"], row["lat"], \
                                                                  row['angle'], int(row["t"]), row['link_id']
                logger.info(
                    "send veh info: t: {}, {}, {}, {}, {}, {}, {}, {}, {}".format(t,r_veh_id, link_id, r_veh_coord_offset_x,
                                                                           r_veh_coord_offset_y, angle, t, x2,
                                                                           y2))
                if r_veh_id == 447:
                    logger.info(
                        "send veh info: t: {}, {}, {}, {}, {}".format(t,r_veh_id, r_veh_coord_offset_x,
                                                                   r_veh_coord_offset_y,
                                                                   angle, ))
                    yield create_point_response(id=str(r_veh_id), lon=y2, lat=x2, x=r_veh_coord_offset_x,
                                                y=r_veh_coord_offset_y, angle=angle, t=t)
            yield create_image_response(image=image_path, t=t)
            if isStop:
                logger.info("t: {}, need stop".format(t))
                yield create_stop_response(isStop=isStop, t=t)
            if complete:
                logger.info("t: {}, game complete".format(t))
                yield create_complete_response(result = True, t = t)

        elif request.HasField("se_info"):
            # self.move_instance = test_game.Trasport((request.se_info.x1, request.se_info.y1),(request.se_info.x2, request.se_info.y2))
            # new_start, new_end = self.move_instance.get_new_od()
            new_start, new_end = self._record_peer(context, request)
            yield create_newse_response(new_start, new_end)
            # whole_vehs, isStop, image_path, complete = self.update_each(direction_tmp=None)
            whole_vehs, isStop, image_path, complete, t = self._record_peer(context, request)

            for index, row in whole_vehs.iterrows():
                r_veh_id, r_veh_coord_offset_x, r_veh_coord_offset_y, x2, y2, angle, t, link_id = int(row["veh_id"]), \
                                              row["lon_offset_utm"], row["lat_offset_utm"], row["lon"], row["lat"], \
                                              row['angle'], int(row["t"]), row['link_id']
                logger.info(
                    "send veh info: t:{}, {}, {}, {}, {}, {}, {}, {}, {}".format(t ,r_veh_id, link_id, r_veh_coord_offset_x,
                                                                           r_veh_coord_offset_y, angle, t, x2, y2))
                if r_veh_id == 447:
                    logger.info(
                        "send veh info: t: {}, {}, {}, {}, {}".format(t, r_veh_id, r_veh_coord_offset_x, r_veh_coord_offset_y,
                                                                   angle, ))
                    yield create_point_response(id=str(r_veh_id), lon=y2, lat=x2, x=r_veh_coord_offset_x,
                                                y=r_veh_coord_offset_y, angle=angle, t=t)
            yield create_image_response(image=image_path, t = t)
            if isStop:
                logger.info("t: {}, need stop".format(t))
                yield create_stop_response(isStop=isStop, t=t)
            if complete:
                logger.info("t: {}, game complete".format(t))
                yield create_complete_response(result = True, t = t)

    def UnSubscribe(self, request, context):
        logger.info("receive: {}".format(request))
        self._peer_set.disconnect(context.peer())
        return move_pb2.Status(state='success')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    move_pb2_grpc.add_MoveServicer_to_server(Move(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

def start():
    serve()

if __name__ == "__main__":
    start()

