from concurrent import futures
import logging

import grpc
from communicate.server import Drive_pb2
from communicate.server import Drive_pb2_grpc
from game import test_game
import random
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def get_distance(start, end):
    x1, y1 = start
    x2, y2 = end
    d = pow(pow(x1-x2,2)+pow(y1-y2,2), 0.5)
    return d


def get_logger(name):
    logging.basicConfig(filename='server.log', filemode='w',
                    format='%(asctime)s - %(message)s', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger(name).addHandler(console)
    return logging.getLogger(name)


isStop_tmp = False
new_find = False
calculate_time = None
calculated_time = None
player_file = open("../output/player.txt", "w", encoding="utf-8")
image_path = ""
start_nid,end_nid = None, None

def get_position(direction, unity_time):
    global data,config,update_data,isStop_tmp,image_path
    if isStop_tmp:
        assert direction!="", "must have direction now"
    data, config, update_data, whole_vehs, isStop_tmp, image_path = test_game.test_each_grpc(unity_time, data, config, update_data, logging, direction, isStop_tmp)
    return whole_vehs, isStop_tmp


class Drive(Drive_pb2_grpc.Drive):
    def GetPosition(self, request,context):
        global calculate_time, calculated_time
        direction = request.op
        unity_time = request.t
        if calculate_time==None:
            calculate_time = unity_time

        if unity_time>=calculate_time:
            calculate_time=unity_time+1
            logger.info("get direction: {}".format(request.op))
            logger.info("start calculating positions...unity_time:{}, calculating_time:{}".format(unity_time,calculate_time-1))
            whole_vehs, isStop_tmp = get_position(direction, unity_time)
            calculated_time = unity_time

            if isinstance(whole_vehs, pd.DataFrame):
                logger.info("end calculating positions, whole_vehs at calculating time: {}, num_vehs: {}".format(
                    calculate_time - 1, len(whole_vehs)))
                if len(whole_vehs)>0:
                    for index, row in whole_vehs.iterrows():
                        r_veh_id, r_veh_coord_offset_x, r_veh_coord_offset_y, x2, y2, angle, t, link_id = int(row["veh_id"]), row["lon_offset_utm"], row["lat_offset_utm"], \
                                                                                                          row["lon"], row["lat"], row['angle'], row["t"], row['link_id']
                        logger.info("send veh info: {}, {}, {}, {}, {}, {}, {}, {}".format(r_veh_id, link_id,r_veh_coord_offset_x, r_veh_coord_offset_y, angle, int(t),x2,y2))
                        if r_veh_id == 447:
                            player_file.write("{}, {}, {}, {}, {}, {}, {}, {} \n".format(int(t), link_id,r_veh_id,r_veh_coord_offset_x,r_veh_coord_offset_y,angle, x2, y2))
                            logger.info("send veh info: {}, {}, {}, {}, {}".format(r_veh_id,r_veh_coord_offset_x,r_veh_coord_offset_y,angle, int(t)))
                        yield Drive_pb2.Point(id=str(r_veh_id), lon = y2, lat = x2, x = r_veh_coord_offset_x, y= r_veh_coord_offset_y, angle = angle, t=int(t))
                else:
                    # calculate_time-=1
                    logger.info("send veh info: id {},x {}, y {}, calculate_time {}".format(-1, -1, -1, int(calculate_time-1)))
                    yield Drive_pb2.Point(id=str(-1), lon=-1, lat=-1, x=-1, y=-1,angle=-1, t=int(calculate_time-1))
            else:
                calculate_time -= 1
                logger.info(
                    "send veh info: id {},x {}, y {}, calculate_time {}".format(-2, -1, -1, int(calculate_time)))
                yield Drive_pb2.Point(id=str(-2), lon=-1, lat=-1, x=-1, y=-1, angle=-1, t=int(calculate_time))

        else:
            # logger.info("get direction: {}, unity_time: {}".format(request.op, unity_time))
            # logger.info("cannot calculate, unity time: {}, calculate time: {}".format(unity_time, calculate_time))
            logger.info("send veh info: id {},x {}, y {}, calculate_time {}".format(-2, -2, -2, int(calculate_time)-1))
            yield Drive_pb2.Point(id=str(-2), lon=-2, lat=-2, x=-2, y=-2, angle=-2, t=int(calculate_time)-1)

    def DriverStop(self, request, context):
        global calculated_time, calculate_time
        if calculated_time and calculate_time and (calculate_time - 1 == calculated_time):

            logger.info("request_time: {}, calculated_time {}, calculate_time {}, vehicle needs stop? {}".format(request.t, calculated_time, calculate_time, isStop_tmp))
            return Drive_pb2.StopInfo(isStop=isStop_tmp, t=calculated_time)
        else:
            return Drive_pb2.StopInfo(isStop=None, t=-1)

    def ChangeImage(self, request, context):
        global calculated_time, calculate_time
        if calculated_time and calculate_time and (calculate_time - 1 == calculated_time):

            logger.info("request_time: {}, calculated_time {}, calculate_time {}, send driver image: {}".format(request.t, calculated_time, calculate_time, image_path))
            return Drive_pb2.ChangeInfo(image=image_path, t=calculated_time)
        else:

            return Drive_pb2.ChangeInfo(image=None, t=-1)

    def SendSEPosition(self, request, context):
        global start_nid, end_nid, calculate_time
        agent_id = 447
        startPoint = (request.x1, request.y1)
        endPoint = (request.x2, request.y2)
        nodes_df = pd.read_csv("../projects/bolinas/network_inputs/bolinas_nodes_sim.csv")
        nodes_df = gpd.GeoDataFrame(nodes_df, crs='epsg:4326', geometry=[Point(x, y) for (x, y) in zip(nodes_df.lon, nodes_df.lat)]).to_crs('epsg:26910')
        nodes_df['x'] = nodes_df['geometry'].apply(lambda x: x.x)
        nodes_df['y'] = nodes_df['geometry'].apply(lambda x: x.y)
        nodes_df = nodes_df[['nid', 'x', 'y', 'osmid']]
        nodes_df['distance_start'] = nodes_df.apply(lambda x: get_distance(startPoint,(x["x"], x["y"])), axis=1)
        nodes_df['distance_end'] = nodes_df.apply(lambda x: get_distance(endPoint,(x['x'],x["y"])), axis=1)
        start_nid = int(nodes_df[nodes_df["distance_start"] == min(nodes_df["distance_start"])].iloc[0]["nid"])
        end_nid = int(nodes_df[nodes_df["distance_end"] == min(nodes_df["distance_end"])].iloc[0]["nid"])

        data["network"].agents[agent_id].origin_nid = start_nid
        data["network"].agents[agent_id].destin_nid = end_nid
        data["network"].agents[agent_id].current_link_start_nid = "vn{}".format(start_nid)
        data["network"].agents[agent_id].current_link_end_nid = start_nid
        logger.info("t: {}, start node id: {}".format(calculate_time, start_nid))
        logger.info("t: {}, end node id: {}".format(calculate_time, end_nid))

        data["network"].agents[agent_id].departure_time = 1
        logger.info("agent' s departure time: {}".format(1))
        return Drive_pb2.SEInfo(state = "success")




def serve():
    global data,config,update_data
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    Drive_pb2_grpc.add_DriveServicer_to_server(Drive(), server)
    server.add_insecure_port('[::]:50051')
    data, config, update_data = test_game.initialize()
    server.start()
    server.wait_for_termination()



if __name__ == "__main__":
    logger = get_logger('root')
    player_file.write("t,link_id,car_id,x,y,heading,lat,lon")
    serve()