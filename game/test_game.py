# %%

import os
import gc
import sys
import time
import json
import random
import logging
import warnings
import numpy as np
import pandas as pd
from pyproj import Transformer
import bisect
pd.set_option('display.max_columns', 500)
import geopandas as gpd
from ctypes import c_double

from shapely.wkt import loads

from model.queue_class import Network
import game.extract_vehicle_locations as extract_vehicle_location
from communicate import server_rpc
print(sys.path)
import ast

root_dir = server_rpc.root_directory
links_list = pd.read_csv(os.path.join(root_dir ,'split_link/links_new.csv'))

class Trasport():
    def __init__(self, origin, destination):
        self.start = origin
        self.end = destination
        self.logger = server_rpc.logger
        self.random_seed = 0
        self.fire_id = '1'  # '1', '2', '3'
        self.comm_id = '1'  # '1', '2', '3'
        self.vphh = 2  # vehicles per household
        self.visitor_cnts = 300
        self.contra_id = '0'
        self.shelter_scen_id = '0'
        self.link_closed_time = 0
        self.closed_mode = 'flame'
        self.agent_id = 447
        self.t = 0
        self.start_nid = None
        self.end_nid = None
        self.start_point = None
        self.end_point = None
        self.scen_nm = "r{}_fire{}_comm{}_vphh{}_vistor{}_contra{}_close{}m{}_shelter{}". \
            format(self.random_seed, self.fire_id, self.comm_id, self.vphh, self.visitor_cnts, self.contra_id,
                   self.link_closed_time, self.closed_mode, self.shelter_scen_id)
        self.initialize()

    def initialize(self,):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.preparation()

    def preparation(self,):
        project_location = '/projects/bolinas'
        network_file_edges = project_location + '/network_inputs/bolinas_edges_sim.csv'
        network_file_nodes = project_location + '/network_inputs/bolinas_nodes_sim.csv'
        network_file_special_nodes = project_location + '/network_inputs/bolinas_special_nodes.json'
        demand_files = [project_location + '/demand_inputs/od_csv/resident_visitor_od_rs{}_commscen{}_vphh{}_visitor{}.csv'.
            format(self.random_seed, self.comm_id, self.vphh, self.visitor_cnts)]

        if self.contra_id == '0':
            cf_files = []
        elif self.contra_id == '1':
            cf_files = [project_location + '/network_inputs/bolinas_contraflow.csv']
        else:
            cf_files = []

        self.logger.info(self.scen_nm)

        ### network
        with open(root_dir + network_file_special_nodes) as special_nodes_file:
            self.special_nodes = json.load(special_nodes_file)
        self.network = Network()
        self.start_nid, self.end_nid, self.start_point, self.end_point = self.network.dataframe_to_network(project_location=project_location, network_file_edges=network_file_edges,
                                     network_file_nodes=network_file_nodes, cf_files=cf_files,
                                     special_nodes=self.special_nodes,
                                     scen_nm=self.scen_nm, startPoint=self.start, endPoint=self.end)
        self.network.add_connectivity()

        ### demand
        self.network.add_demand(demand_files=demand_files)

        for agent_id, agent in self.network.agents.items():
            if (agent_id != self.agent_id):
                self.network.agents[agent_id].departure_time = np.random.randint(1, 4)
            else:
                self.network.agents[agent_id].departure_time = float("inf")
                # network.agents[agent_id].origin_nid =
                # network.agents[agent_id].destin_nid = -1

        self.network.agents[self.agent_id].get_path(self.t, g=self.network.g)
        self.network.agents[self.agent_id].find_next_link(node2link_dict=self.network.node2link_dict)

        #update start and end point
        self.network.agents[self.agent_id].origin_nid = self.start_nid
        self.network.agents[self.agent_id].destin_nid = self.end_nid
        self.network.agents[self.agent_id].current_link_start_nid = "vn{}".format(self.start_nid)
        self.network.agents[self.agent_id].current_link_end_nid = self.start_nid
        self.logger.info("t: {}, start node id: {}".format(self.t, self.start_nid))
        self.logger.info("t: {}, end node id: {}".format(self.t, self.end_nid))


        self.network.agents[self.agent_id].departure_time = 1
        self.logger.info('total numbers of agents taken {}'.format(len(self.network.agents.keys())))

    def get_new_od(self):
        return self.start_point, self.end_point

    def test_each_grpc(self, direction_tmp=None):
        # one_step(t, data, config, update_data, agent_id)
        self.one_step()

        trace_agent = self.network.agents[self.agent_id]
        current_link_id = self.network.node2link_dict[(trace_agent.current_link_start_nid, trace_agent.current_link_end_nid)]
        # alternative links
        alternative_links = []
        checked_outgoing_links = []
        current_link_angle = self.network.links[current_link_id].out_angle
        for alt_link_id in self.network.nodes[trace_agent.current_link_end_nid].outgoing_links:
            alternative_link_angle = (current_link_angle - self.network.links[alt_link_id].in_angle) / 3.14 * 180
            if alternative_link_angle < -180: alternative_link_angle += 360
            if alternative_link_angle > 180: alternative_link_angle -= 360
            alternative_links.append([alt_link_id, self.network.links[alt_link_id].end_nid, alternative_link_angle])
        # print(alternative_links)
        alternative_links_by_direction = {'forward': None, 'left': None, 'right': None, 'back': None}
        # forward
        potential_forward_link = [(forward_link, end_nid, angle) for (forward_link, end_nid, angle) in alternative_links if
                                  (abs(angle - 0) <= 45 and forward_link not in checked_outgoing_links)]
        if (len(potential_forward_link) > 0):
            alternative_links_by_direction['forward'] = min(potential_forward_link, key=lambda l: abs(l[2] - 0))
            checked_outgoing_links.append(alternative_links_by_direction['forward'])

        # left
        potential_left_link = [(left_link, end_nid, angle) for (left_link, end_nid, angle) in alternative_links if
                               (abs(angle - (-90)) <= 45 and (left_link not in checked_outgoing_links))]
        if (len(potential_left_link) > 0):
            alternative_links_by_direction['left'] = min(potential_left_link, key=lambda l: abs(l[2] - (-90)))
            checked_outgoing_links.append(alternative_links_by_direction['left'])

        # right
        potential_right_link = [(right_link, end_nid, angle) for (right_link, end_nid, angle) in alternative_links if
                                (abs(angle - 90) <= 45 and (right_link not in checked_outgoing_links))]
        # (right_link, angle) = min(alternative_links, key=lambda l: abs(l[1] - 90))
        if (len(potential_right_link) > 0):
            alternative_links_by_direction['right'] = min(potential_right_link, key=lambda l: abs(l[2] - 90))
            checked_outgoing_links.append(alternative_links_by_direction['right'])

        # backward
        potential_back_link = [(back_link, end_nid, angle) for (back_link, end_nid, angle) in alternative_links if
                               (abs(angle - 0) > 135 and (back_link not in checked_outgoing_links))]
        if len(potential_back_link) > 0:
            alternative_links_by_direction['back'] = max(potential_back_link, key=lambda l: abs(l[2]))
            checked_outgoing_links.append(alternative_links_by_direction['back'])

        print(alternative_links_by_direction)
        self.logger.info(alternative_links_by_direction)

        agent_next_link_direction = direction_tmp
        valid_links_by_direction = [k for k, v in alternative_links_by_direction.items() if v is not None]

        self.logger.info("agent' s route: {}".format(self.network.agents[self.agent_id].route))

        if agent_next_link_direction in valid_links_by_direction:
            agent_next_link = alternative_links_by_direction[agent_next_link_direction][0]#link_id
            self.logger.info("driver's self decision: {}".format(agent_next_link))
        else:
            agent_next_link = self.network.agents[self.agent_id].next_link #link_id
            self.logger.info("driver's system decision: {}".format(agent_next_link))
            # random.choice(valid_links_by_direction)

        self.logger.info("time: {}, current link is {}, next link is {}".format(self.t, current_link_id, agent_next_link))
        # print('  next link is {}'.format(agent_next_link))
        # logger.info('  next link is {}'.format(agent_next_link))

        # agent_next_link_direction = None
        # close all other directions
        if agent_next_link!=None:
            for [link_id, _, angle_tmp] in alternative_links:
                if link_id != agent_next_link:
                    # print('close {}'.format(link_id))
                    link = self.network.links[link_id]
                    self.network.g.update_edge(link.start_nid, link.end_nid, c_double(1e8))
        else:
            print("next link is none")

        self.network.agents[self.agent_id].get_path(self.t, g = self.network.g)
        self.network.agents[self.agent_id].find_next_link(node2link_dict = self.network.node2link_dict)
        # logger.info("agent' s route: {}".format(network.agents[agent_id].route))
        # print(agent_next_link, network.agents[0].next_link)

        for [link_id, _, angle_tmp] in alternative_links:
            if link_id != agent_next_link:
                link = self.network.links[link_id]
                self.network.g.update_edge(link.start_nid, link.end_nid, c_double(link.fft))

        isComplete = self.calculate_position()

        # extract vehicle positions
        veh_num, vehicle_positions, whole_vehs, isStop = extract_vehicle_location.extract_vehicle_locations(self.network, self.t, self.agent_id, self.logger)
        agent_link = self.network.node2link_dict[(trace_agent.current_link_start_nid, trace_agent.current_link_end_nid)]
        if isComplete:
            self.logger.info("game complete")
        if (type(agent_link)==str and "vl" in agent_link) or (isComplete):
            image_path = ""
        else:
            driver_link_list = ast.literal_eval(links_list["long_points_list"].loc[links_list["eid"]==agent_link].values[0])
            driver_position = (whole_vehs["lon"].loc[whole_vehs["veh_id"] == self.agent_id].values[0],
                               whole_vehs["lat"].loc[whole_vehs["veh_id"] == self.agent_id].values[0])
            image_path_tmp = get_image_rpc(driver_link_list, driver_position)
            # image_path = os.path.join("link_"+str(whole_vehs["link_id"].loc[whole_vehs["veh_id"]==agent_id].values[0]), image_path_tmp)
            image_path = os.path.join(str(whole_vehs["link_id"].loc[whole_vehs["veh_id"] == self.agent_id].values[0]), image_path_tmp)

        self.t+=1

        return whole_vehs, isStop, image_path, isComplete, int(self.t-1)



    def one_step(self):
        ### update link travel time before rerouting
        reroute_freq_dict = {'1': 300, '2': 900, '3': 1800}
        reroute_freq = reroute_freq_dict[self.comm_id]

        ### reset link congested counter
        for link in self.network.links.values():
            link.congested = 0
            if (self.t % 100 == 0):
                link.update_travel_time_by_queue_length(self.network.g)

        # if self.t == 0:
        #     for agent_id, agent in self.network.agents.items():
        #         if (agent_id != self.agent_id):
        #             self.network.agents[agent_id].departure_time = np.random.randint(1, 100)
        #         else:
        #             self.network.agents[agent_id].departure_time = float("inf")
        #             # network.agents[agent_id].origin_nid =
        #             # network.agents[agent_id].destin_nid = -1
        # elif self.t==1:
        #     self.network.agents[self.agent_id].get_path(self.t, g=self.network.g)
        #     self.network.agents[self.agent_id].find_next_link(node2link_dict=self.network.node2link_dict)


        ### agent model
        stopped_agents_list = []
        for agent_id, agent in self.network.agents.items():

            ### first remove arrived vehicles
            if agent.status == 'arrive':
                self.network.agents_stopped[agent_id] = (agent.status, self.t, agent.agent_type)
                stopped_agents_list.append(agent_id)
                continue
            ### find congested vehicles: spent too long in a link
            current_link = self.network.links[self.network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]]
            if (current_link.link_type != 'v') and (agent.current_link_enter_time is not None) and (
                    self.t - agent.current_link_enter_time > 3600 * 0.5):
                current_link.congested += 1
                if (self.shelter_scen_id == '0'):
                    if (self.t - agent.current_link_enter_time > 3600 * 3):
                        agent.status = 'shelter_a1'
                elif self.shelter_scen_id == '1':
                    pass
                else:
                    pass
            ### agents need rerouting
            # initial route
            # if (t == 0) or (t % reroute_freq == agent_id % reroute_freq):
            if (self.t == 0) or (self.t % reroute_freq == 0):
                routing_status = agent.get_path(self.t, g = self.network.g)
                agent.find_next_link(node2link_dict = self.network.node2link_dict)
            agent.load_vehicle(self.t, node2link_dict = self.network.node2link_dict, link_id_dict = self.network.links)
            ### remove passively sheltered vehicles immediately, no need to wait for node model
            if agent.status in ['shelter_p', 'shelter_a1', 'shelter_park']:
                current_link.queue_vehicles = [v for v in current_link.queue_vehicles if v != agent_id]
                current_link.run_vehicles = [v for v in current_link.run_vehicles if v != agent_id]
                self.network.nodes[agent.current_link_end_nid].shelter_counts += 1
                self.network.agents_stopped[agent_id] = (agent.status, self.t, agent.agent_type)
                stopped_agents_list.append(agent_id)
        for agent_id in stopped_agents_list:
            del self.network.agents[agent_id]
        # return network


    def calculate_position(self,):
        ### link model
        ### Each iteration in the link model is not time-consuming. So just keep using one process.

        self.link_model()

        trace_agent = self.network.agents[self.agent_id]
        current_link_id = self.network.node2link_dict[(trace_agent.current_link_start_nid, trace_agent.current_link_end_nid)]
        next_link_id = trace_agent.next_link

        ### node model
        get_end = self.node_model()
        # self.logger.info("t:{}, current link {}' s queue_vehicles: {}, next link {}' s queue_vehicles: {}".format(self.t, current_link_id, self.network.links[current_link_id].queue_vehicles, next_link_id , self.network.links[next_link_id].queue_vehicles))
        # self.logger.info("t:{}, current link {}' s run_vehicles: {}, next link {}' s run_vehicles: {}".format(self.t, current_link_id, self.network.links[current_link_id].run_vehicles, next_link_id, self.network.links[next_link_id].run_vehicles ))

        # stop
        if len(self.network.agents) == 0:
            logging.info("all agents arrive at destinations")
        return get_end

    def link_model(self,):
        # run link model
        for link_id, link in self.network.links.items():
            link.run_link_model(self.t, agent_id_dict=self.network.agents)

    def node_model(self,):
        # only run node model for those with vehicles waiting
        node_ids_to_run = set([link.end_nid for link in self.network.links.values() if len(link.queue_vehicles) > 0])
        get_end = False
        # run node model
        for node_id in node_ids_to_run:
            node = self.network.nodes[node_id]
            _, _, status = node.run_node_model(self.t, node_id_dict=self.network.nodes, link_id_dict=self.network.links, agent_id_dict = self.network.agents,
                                                           node2link_dict = self.network.node2link_dict, special_nodes = self.special_nodes,
                                                           driver_id=self.agent_id, driver_next_link = self.network.agents[self.agent_id].next_link)
            if status:
                get_end = True
        return get_end

def calculate_perpen_point(x0, y0, x1, y1, x2, y2):
    k = -((x1-x0)*(x2-x1)+(y1-y0)*(y2-y1))/(pow(x2-x1,2)+pow(y2-y1,2))
    xn = k*(x2-x1)+x1
    yn = k*(y2-y1)+y1
    return xn,yn

def get_image_rpc(driver_link_list, driver_position):
    find = False
    for i in range(len(driver_link_list)-1):
        x1, y1 = driver_link_list[i]
        x2, y2 = driver_link_list[i+1]
        x0, y0 = driver_position
        perpe_point = calculate_perpen_point(x0, y0, x1, y1, x2, y2)
        if min(x1,x2)<=perpe_point[0]<max(x1,x2) and min(y1, y2) <=perpe_point[1]< max(y1,y2):
            find = True
            # return "p{}_{}_{}.jpg".format(i,x1,y1)
            return "p{}_{}_{}__final.jpg".format(i, x1, y1)

    if not find:
        print("image not find")
    return ""




